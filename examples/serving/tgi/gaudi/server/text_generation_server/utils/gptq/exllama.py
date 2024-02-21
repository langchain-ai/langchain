import torch
from exllama_kernels import make_q4, q4_matmul, prepare_buffers, set_tuning_params

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(
        qweight, qzeros, scales, g_idx if g_idx is not None else none_tensor, device
    )


def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

    q4_matmul(x, q4, output)

    return output.view(outshape)


MAX_DQ = 1
MAX_INNER = 1
ACT_ORDER = False
DEVICE = None

TEMP_STATE = None
TEMP_DQ = None


def set_device(device):
    global DEVICE
    DEVICE = device


def create_exllama_buffers():
    global MAX_DQ, MAX_INNER, ACT_ORDER, DEVICE, TEMP_STATE, TEMP_DQ

    assert DEVICE is not None, "call set_device first"

    if ACT_ORDER:
        # TODO: this should be set to rust side `max_total_tokens`, but TGI
        # does not offer an API to expose this variable to python, as this variable
        # is handled by the client but it appears the model is initialized by the server.
        # An alternative could be to initialize the buffers during warmup.
        # Dummy
        max_total_tokens = 2048
    else:
        max_total_tokens = 1

    # This temp_state buffer is required to reorder X in the act-order case.
    temp_state = torch.zeros(
        (max_total_tokens, MAX_INNER), dtype=torch.float16, device=DEVICE
    )
    temp_dq = torch.zeros((1, MAX_DQ), dtype=torch.float16, device=DEVICE)

    # This temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
    prepare_buffers(DEVICE, temp_state, temp_dq)

    matmul_recons_thd = 8
    matmul_fused_remap = False
    matmul_no_half2 = False
    set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

    TEMP_STATE, TEMP_DQ = temp_state, temp_dq


class Ex4bitLinear(torch.nn.Module):
    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        global MAX_DQ, MAX_INNER, ACT_ORDER, DEVICE
        assert bits == 4

        self.device = qweight.device
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx.cpu() if g_idx is not None else None
        self.bias = bias if bias is not None else None

        if self.g_idx is not None and (
            (self.g_idx == 0).all()
            or torch.equal(
                g_idx.cpu(),
                torch.tensor(
                    [i // groupsize for i in range(g_idx.shape[0])], dtype=torch.int32
                ),
            )
        ):
            self.empty_g_idx = True
            self.g_idx = None

        assert self.device.type == "cuda"
        assert self.device.index is not None

        self.q4 = ext_make_q4(
            self.qweight, self.qzeros, self.scales, self.g_idx, self.device.index
        )

        self.height = qweight.shape[0] * 8
        self.width = qweight.shape[1]

        # Infer groupsize from height of qzeros
        self.groupsize = None
        if self.qzeros.shape[0] > 1:
            self.groupsize = (self.qweight.shape[0] * 8) // (self.qzeros.shape[0])

        if self.groupsize is not None:
            assert groupsize == self.groupsize

        # Handle act-order matrix
        if self.g_idx is not None:
            if self.groupsize is None:
                raise ValueError("Found group index but no groupsize. What do?")
            self.act_order = True
        else:
            self.act_order = False

        DEVICE = self.qweight.device

        MAX_DQ = max(MAX_DQ, self.qweight.numel() * 8)

        if self.act_order:
            MAX_INNER = max(MAX_INNER, self.height, self.width)

            ACT_ORDER = True

    def forward(self, x):
        out = ext_q4_matmul(x, self.q4, self.width)

        if self.bias is not None:
            out.add_(self.bias)
        return out
