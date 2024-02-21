import math
import torch

from typing import Optional, List, Tuple

BLOCK_SIZE: int = 16
# Will be set in warmup
CACHE_MANAGER: Optional["CacheManager"] = None


class CacheManager:
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        repeat_slots: bool,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.block_size = BLOCK_SIZE
        self.num_blocks = num_blocks
        self.repeat_slots = repeat_slots

        element_size = torch.tensor([], dtype=dtype).element_size()
        x = self.block_size // element_size

        self.kv_cache = [
            (
                torch.empty(
                    (num_blocks, num_heads, head_size // x, self.block_size, x),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, num_heads, head_size, self.block_size),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(num_layers)
        ]
        self.free_block_mask = torch.ones(num_blocks, dtype=torch.int32, device="cpu")
        self.slots = torch.arange(
            0, num_blocks * self.block_size, dtype=torch.int32
        ).view(num_blocks, self.block_size)

    def allocate(
        self,
        needed_blocks_slots: List[Tuple[int, int]],
        blocks: int,
        max_blocks: int,
        device: torch.device,
    ):
        # Get free blocks indices by finding values in mask that are not set to 0
        free_block_indices = self.free_block_mask.nonzero()
        assert (
            len(free_block_indices) >= blocks
        ), f"Out of available cache blocks: asked {blocks}, only {len(free_block_indices)} free blocks"

        # Slice by the number of required blocks
        block_indices = free_block_indices[:blocks]
        block_indices = block_indices.flatten()

        # Padded block tables
        block_tables_tensor = torch.zeros(
            (len(needed_blocks_slots), max_blocks), dtype=torch.int32
        )

        # Allocate paged attention blocks
        cumulative_blocks = 0
        slots = []
        block_tables = []
        for i, (needed_blocks, needed_slots) in enumerate(needed_blocks_slots):
            # Get allocated blocks for this sequence
            allocated_blocks = block_indices[
                cumulative_blocks : cumulative_blocks + needed_blocks
            ]
            # Get slots for the allocated blocks
            all_slots = self.slots[allocated_blocks].flatten()

            # Repeat slots in the case of context sliding window
            if needed_slots > len(all_slots) and self.repeat_slots:
                repeats = math.ceil(needed_slots / len(all_slots))
                all_slots = all_slots.repeat(repeats)

            allocated_slots = all_slots[:needed_slots]

            slots.append(allocated_slots)
            block_tables.append(allocated_blocks.tolist())
            block_tables_tensor[i, :needed_blocks] = allocated_blocks
            cumulative_blocks += needed_blocks

        block_tables = block_tables
        block_tables_tensor = block_tables_tensor.to(device)
        slots = torch.concat(slots).to(device)

        # Allocate the required number of blocks by setting the mask to 0
        self.free_block_mask[block_indices] = 0

        return block_tables, block_tables_tensor, slots

    def free(self, block_indices: Optional[List[int]]):
        if block_indices is not None and block_indices:
            # Reset mask
            self.free_block_mask[block_indices] = 1


def set_cache_manager(
    num_blocks: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    repeat_slots: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> CacheManager:
    global CACHE_MANAGER
    if CACHE_MANAGER is not None:
        del CACHE_MANAGER
        torch.cuda.empty_cache()

    CACHE_MANAGER = CacheManager(
        num_blocks, num_layers, num_heads, head_size, repeat_slots, dtype, device
    )
    return CACHE_MANAGER


def get_cache_manager() -> CacheManager:
    global CACHE_MANAGER
    if CACHE_MANAGER is None:
        raise RuntimeError("cache manager was not initialized")

    return CACHE_MANAGER
