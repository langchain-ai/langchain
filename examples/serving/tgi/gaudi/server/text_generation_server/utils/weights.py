import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from safetensors import safe_open, SafetensorError
import torch
from loguru import logger
from huggingface_hub import hf_hub_download
import json


class Weights:
    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
        aliases: Optional[Dict[str, List[str]]] = None,
        prefix: Optional[str] = None
    ):
        routing = {}
        for filename in filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self.prefix = prefix
        self._handles = {}

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):

        names = [tensor_name]
        if self.prefix is not None:
            prefixed = f"{self.prefix}.{tensor_name}"
            names.append(prefixed)
        for name in names:
            filename = self.routing.get(name, None)
            if filename is not None:
                return str(filename), name

            aliases = self.aliases.get(name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
        raise RuntimeError(f"weight {tensor_name} does not exist")

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str, to_device=True):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        if to_device:
            tensor = tensor.to(device=self.device)
        return tensor

    def get_partial_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        size = slice_.get_shape()[dim]
        block_size = size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype != torch.int32:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        assert (
            size % world_size == 0
        ), f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        return self.get_partial_sharded(tensor_name, dim)

    def _get_qweight(self, name: str):
        slice_ = self._get_slice(name)
        total_size = slice_.get_shape()[1]
        assert total_size % 3 == 0, "Prepacked quantized qkv is not divisible by 3"
        single_size = total_size // 3
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        assert (
            single_size % world_size == 0
        ), f"Prepacked quantized qkv cannot be sharded across {world_size} shards"
        block_size = single_size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size
        q = slice_[:, start:stop]
        k = slice_[:, start + single_size : stop + single_size]
        v = slice_[:, start + 2 * single_size : stop + 2 * single_size]
        weight = torch.cat([q, k, v], dim=1)
        weight = weight.to(device=self.device)
        return weight

    def get_weights_col_packed_qkv(self, prefix: str, quantize: str):
        """
        Highly specific when the underlying tensor is a simple cat of Q,K,V instead of being
        already alternating Q,K,V within the main tensor
        """
        if quantize in ["gptq", "awq"]:
            try:
                qweight = self._get_qweight(f"{prefix}.qweight")
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight, make sure the model is already quantized."
                )

            qzeros = self._get_qweight(f"{prefix}.qzeros")
            scales = self._get_qweight(f"{prefix}.scales")
            scales = scales.to(dtype=self.dtype)
            if quantize == "gptq":
                g_idx = self.get_tensor(f"{prefix}.g_idx")
            else:
                g_idx = None

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        else:
            slice_ = self._get_slice(f"{prefix}.weight")
            total_size = slice_.get_shape()[0]
            assert total_size % 3 == 0, "Prepacked qkv is not divisible by 3"
            single_size = total_size // 3
            world_size = self.process_group.size()
            rank = self.process_group.rank()

            assert (
                single_size % world_size == 0
            ), f"Prepacked qkv cannot be sharded across {world_size} shards"
            block_size = single_size // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
            q = slice_[start:stop]
            k = slice_[start + single_size : stop + single_size]
            v = slice_[start + 2 * single_size : stop + 2 * single_size]
            weight = torch.cat([q, k, v], dim=0)
            weight = weight.to(device=self.device)
            weight = weight.to(dtype=self.dtype)
        return weight

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int):
        if quantize in ["gptq", "awq"]:
            try:
                qweight = torch.cat(
                    [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight, make sure the model is already quantized"
                )

            qzeros = torch.cat(
                [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
            )
            scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
            )

            if quantize == "gptq":
                w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
                for w2 in w[1:]:
                    torch.testing.assert_close(w2, w[0])
                g_idx = w[0]
            else:
                g_idx = None

            bits, groupsize = self._get_gptq_params()
            from text_generation_server.utils.layers import HAS_EXLLAMA
            use_exllama = bits==4  and HAS_EXLLAMA and quantize == "gptq"
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
        else:
            w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
            weight = torch.cat(w, dim=dim)
        return weight

    def get_tensor_shard(self, var, dim):
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        block_size = var.size()[dim] // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size
        if dim == 0:
            tensor = var[start:stop]
        elif dim == 1:
            tensor = var[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_multi_weights_row(self, prefix: str, quantize: str):
        if quantize == "gptq":
            use_exllama = True
            bits, groupsize = self._get_gptq_params()

            if bits != 4:
                use_exllama = False

            if self.process_group.size() > 1:
                g_idx = self.get_tensor(f"{prefix}.g_idx")
                if g_idx is not None:
                    if (
                        not torch.equal(
                            g_idx.cpu(),
                            torch.tensor(
                                [i // groupsize for i in range(g_idx.shape[0])],
                                dtype=torch.int32,
                            ),
                        )
                        and not (g_idx == 0).all()
                    ):
                        # Exllama implementation does not support row tensor parallelism with act-order, as
                        # it would require to reorder input activations that are split unto several GPUs
                        use_exllama = False

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            from text_generation_server.utils.layers import HAS_EXLLAMA, CAN_EXLLAMA

            if use_exllama:
                if not HAS_EXLLAMA:
                    if CAN_EXLLAMA:
                        logger.warning(
                            "Exllama GPTQ cuda kernels (which are faster) could have been used, but are not currently installed, try using BUILD_EXTENSIONS=True"
                        )
                    use_exllama = False
                else:
                    logger.info(f"Using exllama kernels v{HAS_EXLLAMA}")

            if use_exllama:
                qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                scales = self.get_sharded(f"{prefix}.scales", dim=0)
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim= 0)
                g_idx = g_idx - g_idx[0]
            else:
                # The triton kernel reorders the scales/zero points instead of the weight/activation.
                # Thus, each rank needs the full qzeros/scales.
                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
        elif quantize == "awq":
            bits, groupsize = self._get_gptq_params()

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `awq` weight, make sure the model is already quantized"
                )

            qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
            scales = self.get_sharded(f"{prefix}.scales", dim=0)
            g_idx = None
            use_exllama = False

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
        return weight

    def _get_gptq_params(self) -> Tuple[int, int]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
            except Exception:
                raise e

        return bits, groupsize

    def _set_gptq_params(self, model_id):
        filename = "config.json"
        try:
            if os.path.exists(os.path.join(model_id, filename)):
                filename = os.path.join(model_id, filename)
            else:
                filename = hf_hub_download(model_id, filename=filename)
            with open(filename, "r") as f:
                data = json.load(f)
            self.gptq_bits = data["quantization_config"]["bits"]
            self.gptq_groupsize = data["quantization_config"]["group_size"]
        except Exception:
            filename = "quantize_config.json"
            try:
                if os.path.exists(os.path.join(model_id, filename)):
                    filename = os.path.join(model_id, filename)
                else:
                    filename = hf_hub_download(model_id, filename=filename)
                with open(filename, "r") as f:
                    data = json.load(f)
                self.gptq_bits = data["bits"]
                self.gptq_groupsize = data["group_size"]
            except Exception:
                filename = "quant_config.json"
                try:
                    if os.path.exists(os.path.join(model_id, filename)):
                        filename = os.path.join(model_id, filename)
                    else:
                        filename = hf_hub_download(model_id, filename=filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    self.gptq_bits = data["w_bit"]
                    self.gptq_groupsize = data["q_group_size"]
                except Exception:
                    pass
