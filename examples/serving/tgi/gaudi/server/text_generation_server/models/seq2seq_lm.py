from text_generation_server.utils.tokens import batch_top_tokens
import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict

from text_generation_server.models import Model
from text_generation_server.models.types import (
    GeneratedText,
    Batch,
    Generation,
    PrefillTokens,
    TopTokens,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling

tracer = trace.get_tracer(__name__)


@dataclass
class Seq2SeqLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Encoder values
    input_ids: Optional[torch.Tensor]
    attention_mask: torch.Tensor

    # Decoder values
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: Optional[torch.Tensor]
    encoder_last_hidden_state: Optional[torch.Tensor]

    # All tokens
    all_decoder_input_ids: List[torch.Tensor]

    # Seq2SeqLM keeps track of both encoder and decoder attention keys and values
    past_key_values: Optional[List[Tuple]]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    decoder_input_lengths: List[int]
    prefix_offsets: List[int]
    read_offsets: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    # Metadata used for padding
    max_input_length: int
    max_decoder_input_length: int
    padding_right_offset: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    def to_pb(self) -> generate_pb2.CachedBatch:
        """Convert a Seq2SeqLMBatch to a text_generation_server.v1.CachedBatch protobuf"""
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Seq2SeqLMBatch":
        """Convert a text_generation_server.v1.Batch protobuf to a Seq2SeqLMBatch"""
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        decoder_input_lengths = []
        prefix_offsets = []
        read_offsets = []
        requests_idx_mapping = {}

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        max_decode_tokens = 0
        for i, r in enumerate(pb.requests):
            inputs.append(r.inputs)
            requests_idx_mapping[r.id] = i
            decoder_input_lengths.append(1)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
            max_truncation = max(max_truncation, r.truncate)
            max_decode_tokens += stopping_criteria.max_new_tokens
            padding_right_offset = max(
                padding_right_offset, stopping_criteria.max_new_tokens
            )

        # Tokenize batch
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        ).to(device)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        # Decoder sequence only contains the bos_token
        decoder_input_ids = (
            torch.tensor(tokenizer.bos_token_id, device=device)
            .repeat(len(pb.requests))
            .view(-1, 1)
        )
        for _ in pb.requests:
            prefix_offsets.append(0)
            read_offsets.append(1)
        all_decoder_input_ids = decoder_input_ids.view(-1).split(1)
        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        max_tokens = len(inputs) * (max_input_length + max_decode_tokens)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            all_decoder_input_ids=list(all_decoder_input_ids),
            decoder_attention_mask=None,
            encoder_last_hidden_state=None,
            past_key_values=None,
            input_lengths=input_lengths.tolist(),
            decoder_input_lengths=decoder_input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length.item(),
            max_decoder_input_length=1,
            padding_right_offset=padding_right_offset,
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> Optional["Seq2SeqLMBatch"]:
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        requests = []
        input_lengths = []
        decoder_input_lengths = []
        prefix_offsets = []
        read_offsets = []

        all_decoder_input_ids = []

        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []

        max_input_length = 0
        max_decoder_input_length = 0
        padding_right_offset = 0

        total_remaining_decode_tokens = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            keep_indices.append(idx)

            requests.append(self.requests[idx])
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            all_decoder_input_ids.append(self.all_decoder_input_ids[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)

            request_decoder_input_length = self.decoder_input_lengths[idx]
            decoder_input_lengths.append(request_decoder_input_length)
            max_decoder_input_length = max(
                max_decoder_input_length, request_decoder_input_length
            )

            next_token_choosers.append(self.next_token_choosers[idx])
            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(self.top_n_tokens[idx])
            remaining_decode_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )
            total_remaining_decode_tokens += remaining_decode_tokens
            padding_right_offset = max(padding_right_offset, remaining_decode_tokens)

        # Apply indices to input_ids, attention mask, past key values and other items that need to be cached
        self.decoder_input_ids = self.decoder_input_ids[keep_indices]
        self.attention_mask = self.attention_mask[keep_indices, -max_input_length:]
        if self.decoder_attention_mask is not None:
            self.decoder_attention_mask = self.decoder_attention_mask[
                keep_indices,
                -(self.padding_right_offset + max_decoder_input_length) : (
                    self.decoder_attention_mask.shape[1] - self.padding_right_offset
                )
                + padding_right_offset,
            ]

        self.encoder_last_hidden_state = self.encoder_last_hidden_state[
            keep_indices, -max_input_length:
        ]

        # Ensure that past_key_values tensors can be updated in-place
        if type(self.past_key_values[0]) == tuple:
            self.past_key_values = [
                [t for t in layer] for layer in self.past_key_values
            ]

        decoder_past_seq_len = max_decoder_input_length - 1
        for layer in self.past_key_values:
            layer[0] = layer[0][keep_indices, :, -decoder_past_seq_len:]
            layer[1] = layer[1][keep_indices, :, -decoder_past_seq_len:]
            layer[2] = layer[2][keep_indices, :, -max_input_length:]
            layer[3] = layer[3][keep_indices, :, -max_input_length:]

        top_n_tokens_tensor = self.top_n_tokens_tensor[keep_indices]
        max_tokens = (
            len(request_ids) * (max_input_length + max_decoder_input_length)
            + remaining_decode_tokens
        )

        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping
        self.input_ids = None
        self.all_decoder_input_ids = all_decoder_input_ids
        self.input_lengths = input_lengths
        self.decoder_input_lengths = decoder_input_lengths
        self.prefix_offsets = prefix_offsets
        self.read_offsets = read_offsets
        self.next_token_choosers = next_token_choosers
        self.stopping_criterias = stopping_criterias
        self.top_n_tokens = top_n_tokens
        self.top_n_tokens_tensor = top_n_tokens_tensor
        self.max_input_length = max_input_length
        self.max_decoder_input_length = max_decoder_input_length
        self.padding_right_offset = padding_right_offset
        self.max_tokens = max_tokens

        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["Seq2SeqLMBatch"]) -> "Seq2SeqLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        max_decoder_input_length = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            max_decoder_input_length = max(
                max_decoder_input_length, batch.max_decoder_input_length
            )
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        all_decoder_input_ids = []
        input_lengths = []
        decoder_input_lengths = []
        prefix_offsets = []
        read_offsets = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        max_tokens = 0

        # Batch tensors
        attention_mask = None
        decoder_input_ids = None
        decoder_attention_mask = None
        encoder_last_hidden_state = None
        top_n_tokens_tensor = None
        past_key_values = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0

        for i, batch in enumerate(batches):
            # Extend all list attributes
            requests.extend(batch.requests)
            all_decoder_input_ids.extend(batch.all_decoder_input_ids)
            input_lengths.extend(batch.input_lengths)
            decoder_input_lengths.extend(batch.decoder_input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)
            top_n_tokens.extend(batch.top_n_tokens)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.encoder_last_hidden_state is None:
                raise ValueError("Batch encoder_last_hidden_state cannot be None")

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length),
                )
            # Copy to correct indices
            attention_mask[
                start_index:end_index, -batch.max_input_length :
            ] = batch.attention_mask[:, -batch.max_input_length :]

            # Create padded tensor
            if decoder_input_ids is None:
                decoder_input_ids = batch.decoder_input_ids.new_zeros(
                    (total_batch_size, 1),
                )
            # Copy to correct indices
            decoder_input_ids[start_index:end_index] = batch.decoder_input_ids

            # Create padded tensor
            if decoder_attention_mask is None:
                # As decoder_attention_mask might not exist, we use `batch.attention_mask` for device here
                decoder_attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_decoder_input_length + padding_right_offset),
                )
            # If the decoder mask does not exist yet, all generations started at the same time and we never concatenated
            # this batch. All generations are of length `batch.max_decoder_input_length`.
            left_offset = max_decoder_input_length - batch.max_decoder_input_length
            if batch.decoder_attention_mask is None:
                decoder_attention_mask[
                    start_index:end_index,
                    left_offset:-padding_right_offset,
                ] = 1
            # If it exists, we need to index
            else:
                batch_left_offset = (
                    batch.decoder_attention_mask.shape[1]
                    - batch.max_decoder_input_length
                    - batch.padding_right_offset
                )
                decoder_attention_mask[
                    start_index:end_index,
                    left_offset:-padding_right_offset,
                ] = batch.decoder_attention_mask[
                    :,
                    batch_left_offset : -batch.padding_right_offset,
                ]

            # Create padded tensor
            if encoder_last_hidden_state is None:
                encoder_last_hidden_state = batch.encoder_last_hidden_state.new_zeros(
                    (
                        total_batch_size,
                        max_input_length,
                        batch.encoder_last_hidden_state.shape[-1],
                    ),
                )

            if top_n_tokens_tensor is None:
                top_n_tokens_tensor = batches[0].top_n_tokens_tensor.new_zeros(
                    total_batch_size,
                )
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor

            # Copy to correct indices
            encoder_last_hidden_state[
                start_index:end_index, -batch.max_input_length :, :
            ] = batch.encoder_last_hidden_state[:, -batch.max_input_length :, :]
            batch.encoder_last_hidden_state = None

            # Ensure that we can update tensors in-place
            if type(batch.past_key_values[0]) == tuple:
                batch.past_key_values = [
                    [t for t in layer] for layer in batch.past_key_values
                ]

            # Add eventual padding tokens that were added while concatenating
            max_tokens += batch.max_tokens + (
                max_input_length
                - batch.max_input_length
                + max_decoder_input_length
                - batch.max_decoder_input_length
            ) * len(batch)

            start_index = end_index

        # Determine shapes for new past kv tensors
        first_past_kvs = batches[0].past_key_values
        _, num_heads, _, head_dim = first_past_kvs[0][0].shape

        padded_dec_t_shape = (
            total_batch_size,
            num_heads,
            (max_decoder_input_length - 1),
            head_dim,
        )

        padded_enc_t_shape = (
            total_batch_size,
            num_heads,
            max_input_length,
            head_dim,
        )

        # Iterate over attention layers
        for j in range(len(first_past_kvs)):
            past_key_values.append([])

            # Decoder past
            for k in range(0, 2):
                # Initialize tensors
                padded_past_values = first_past_kvs[j][k].new_zeros(padded_dec_t_shape)
                past_key_values[j].append(padded_past_values)

                start_index = 0
                for batch in batches:
                    t = batch.past_key_values[j][k]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][k] = None
                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the past keys and values to remove the padding from previous batches
                    past_seq_len = batch.max_decoder_input_length - 1
                    padded_past_values[start_index:end_index, :, -past_seq_len:, :] = t[
                        :, :, -past_seq_len:, :
                    ]
                    del t

                    start_index = end_index

            # Encoder past
            for k in range(2, 4):
                # Initialize tensors
                padded_past_values = first_past_kvs[j][k].new_zeros(padded_enc_t_shape)
                past_key_values[j].append(padded_past_values)

                start_index = 0
                for batch in batches:
                    t = batch.past_key_values[j][k]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][k] = None
                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the past keys and values to remove the padding from previous batches
                    padded_past_values[
                        start_index:end_index, :, -batch.max_input_length :, :
                    ] = t[:, :, -batch.max_input_length :, :]
                    del t

                    start_index = end_index

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            all_decoder_input_ids=all_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_last_hidden_state=encoder_last_hidden_state,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            decoder_input_lengths=decoder_input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length,
            max_decoder_input_length=max_decoder_input_length,
            padding_right_offset=padding_right_offset,
            max_tokens=max_tokens,
        )

    def __len__(self):
        return len(self.requests)


class Seq2SeqLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else None,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        )
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            model = model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.bos_token_id = model.config.decoder_start_token_id

        super(Seq2SeqLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[Seq2SeqLMBatch]:
        return Seq2SeqLMBatch

    def decode(self, decoder_ids: List[int]) -> str:
        return self.tokenizer.decode(
            decoder_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask: Optional,
        encoder_last_hidden_state: Optional,
        past_key_values: Optional = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_last_hidden_state,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return (
            outputs.logits,
            outputs.encoder_last_hidden_state,
            outputs.past_key_values,
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: Seq2SeqLMBatch
    ) -> Tuple[List[Generation], Optional[Seq2SeqLMBatch]]:
        if batch.decoder_attention_mask is not None:
            # slice to the correct shape
            decoder_attention_mask = batch.decoder_attention_mask[
                :, : -batch.padding_right_offset
            ]
        else:
            decoder_attention_mask = None

        # Wrap `encoder_last_hidden_state` because for some reason, Transformers does a `encoder_last_hidden_state[0]`
        # internally...
        if batch.encoder_last_hidden_state is not None:
            encoder_last_hidden_state = [batch.encoder_last_hidden_state]
        else:
            encoder_last_hidden_state = None

        logits, encoder_last_hidden_state, past = self.forward(
            batch.input_ids,
            batch.attention_mask,
            batch.decoder_input_ids,
            decoder_attention_mask,
            encoder_last_hidden_state,
            batch.past_key_values,
        )

        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1], -1),
        )

        # Finished requests
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.decoder_input_lengths,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_decoder_input_ids,
            batch.top_n_tokens,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            decoder_input_length,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_decoder_input_ids,
            top_n_tokens,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_decoder_input_ids.view(1, -1), logits[-1:, :]
            )

            # Append next token to decoder tokens
            all_decoder_input_ids = torch.cat(
                [all_decoder_input_ids, next_token_id.squeeze(1)]
            )
            new_decoder_input_length = decoder_input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_decoder_input_ids, prefix_offset, read_offset
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(next_token_id, next_token_text)

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Slice with decoder_input_length to remove padding
                    # Decode all tokens
                    output_text, _, _ = self.decode_token(
                        all_decoder_input_ids,
                        prefix_offset=len(all_decoder_input_ids)
                        - decoder_input_length
                        - 1,
                        read_offset=len(all_decoder_input_ids) - decoder_input_length,
                        skip_special_tokens=True,
                    )

                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(
                        output_text, stopping_criteria.current_tokens, reason, seed
                    )
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    prefill_tokens = PrefillTokens(
                        [self.tokenizer.bos_token_id],
                        [float("nan")],
                        [self.tokenizer.bos_token],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    toptoken_texts = self.tokenizer.batch_decode(
                        top_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    special_toptokens = [
                        token_id in self.all_special_ids for token_id in top_token_ids
                    ]
                    top_tokens = TopTokens(
                        top_token_ids,
                        top_token_logprobs,
                        toptoken_texts,
                        special_toptokens,
                    )
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    next_token_id_squeezed,
                    next_token_logprob,
                    next_token_text,
                    next_token_id_squeezed.item() in self.all_special_ids,
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            # Update values
            batch.decoder_input_ids[i] = next_token_id
            batch.all_decoder_input_ids[i] = all_decoder_input_ids
            batch.input_lengths[i] = input_length
            batch.decoder_input_lengths[i] = new_decoder_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, input_length)
            batch.max_decoder_input_length = max(
                batch.max_decoder_input_length, new_decoder_input_length
            )

        # We finished all generations in the batch; there is no next batch
        if stopped:
            return generations, None

        # We don't need input_ids after the prefill forward
        batch.input_ids = None
        batch.encoder_last_hidden_state = encoder_last_hidden_state
        batch.past_key_values = past
        # Update decoder_attention_mask as we added a new token to input_ids
        if batch.decoder_attention_mask is not None:
            batch.decoder_attention_mask[:, -batch.padding_right_offset] = 1
        batch.padding_right_offset -= 1

        return generations, batch
