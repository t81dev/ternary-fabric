import torch
import torch.nn as nn
import numpy as np
import pytfmbs
import struct
from typing import Optional
from .constants import (
    SRAM_BANK_A_OFFSET, SRAM_BANK_B_OFFSET, SRAM_TILE_STRIDE,
    KERNEL_T_GEMM, HINT_ZERO_SKIP, LANES_PER_TILE, MAX_TILES
)

def pack_pt5_numpy(trits):
    """
    Packs trits (-1, 0, 1) into PT-5 format (5 trits per byte).
    trits: numpy array of shape (N,)
    returns: numpy array of uint8
    """
    trits = np.asanyarray(trits)
    # Ensure trits are in {-1, 0, 1}
    trits = np.clip(trits, -1, 1).astype(np.int8)

    # Pad to multiple of 5
    padding = (5 - (len(trits) % 5)) % 5
    if padding > 0:
        trits = np.concatenate([trits, np.zeros(padding, dtype=np.int8)])

    trits_reshaped = trits.reshape(-1, 5)
    powers = 3 ** np.arange(5)
    packed = np.sum((trits_reshaped + 1) * powers, axis=1).astype(np.uint8)
    return packed

def pack_gemv_input(x):
    """
    Packs a ternary input vector x for GEMV.
    Each element x[d] is duplicated for all lanes to fill a multi-lane frame.
    """
    x = np.asanyarray(x).astype(np.int8)
    depth = len(x)

    # Repeated x: [depth, LANES_PER_TILE]
    x_expanded = np.repeat(x[:, np.newaxis], LANES_PER_TILE, axis=1)

    # Reshape to [depth * (LANES_PER_TILE/5), 5]
    x_reshaped = x_expanded.reshape(-1, 5)

    powers = 3 ** np.arange(5)
    packed_bytes = np.sum((x_reshaped + 1) * powers, axis=1).astype(np.uint8)

    # Now we have depth * 3 bytes. We need to add 1 byte of padding after every 3 bytes for 32-bit alignment.
    packed_32 = np.zeros((depth, 4), dtype=np.uint8)
    packed_32[:, 0:3] = packed_bytes.reshape(depth, 3)

    return packed_32.tobytes()

def pack_gemv_weights(w_ternary):
    """
    Packs ternary weights [out_features, in_features] for T-GEMV.
    Returns a list of bytes, one for each tile (up to MAX_TILES).
    """
    out_features, in_features = w_ternary.shape
    w_t = w_ternary.T # [in_features, out_features]

    tile_data = []
    for t in range(MAX_TILES):
        tile_out_start = t * LANES_PER_TILE
        tile_out_end = (t + 1) * LANES_PER_TILE

        # Extract weights for this tile's lanes
        if tile_out_start < out_features:
            tile_w = w_t[:, tile_out_start:min(tile_out_end, out_features)]
            # Pad lanes to LANES_PER_TILE
            if tile_w.shape[1] < LANES_PER_TILE:
                tile_w = np.pad(tile_w, ((0, 0), (0, LANES_PER_TILE - tile_w.shape[1])))

            # tile_w: [in_features, LANES_PER_TILE]
            tile_w_reshaped = tile_w.reshape(-1, 5)
            powers = 3 ** np.arange(5)
            packed_bytes = np.sum((tile_w_reshaped + 1) * powers, axis=1).astype(np.uint8)

            packed_32 = np.zeros((in_features, 4), dtype=np.uint8)
            packed_32[:, 0:3] = packed_bytes.reshape(in_features, 3)
            tile_data.append(packed_32.tobytes())
        else:
            tile_data.append(None)

    return tile_data

class TFMBSLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fabric, weight_addr, bias, in_features, out_features, tile_mask, next_layer=None):
        # input: [batch, in_features]
        # input must be quantized to {-1, 0, 1} for the fabric
        x_ternary = torch.sign(input).to(torch.int8).cpu().numpy()

        batch_size = x_ternary.shape[0]
        outputs = []

        for i in range(batch_size):
            packed_x = pack_gemv_input(x_ternary[i])

            # Load input to Bank B of all active tiles
            for t in range(MAX_TILES):
                if tile_mask & (1 << t):
                    fabric.load(SRAM_BANK_B_OFFSET + t * SRAM_TILE_STRIDE, packed_x)

            # Run asynchronously
            tfd = {
                "base_addr": weight_addr,
                "depth": in_features,
                "lane_count": LANES_PER_TILE,
                "tile_mask": tile_mask,
                "exec_hints": KERNEL_T_GEMM | HINT_ZERO_SKIP,
            }
            fabric.submit(tfd)

            # PIPELINING: Overlap weight loading for the next layer with current execution
            if i == batch_size - 1 and next_layer is not None:
                 next_layer.prefetch()

            # Explicit wait for completion
            fabric.wait()

            # Results
            all_res = fabric.results(-1)
            outputs.append(torch.tensor(all_res[:out_features], dtype=torch.float32))

        output = torch.stack(outputs).to(input.device)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass currently returns the gradient with respect to input.
        Weights are not updated through the fabric in this version (Inference only).
        """
        return grad_output, None, None, None, None, None, None, None

class TFMBSLinear(nn.Module):
    """
    TFMBS-accelerated Linear layer.
    Automatically quantizes weights and offloads GEMV to Ternary Fabric.
    """
    def __init__(self, in_features, out_features, fabric=None, bias=True, weight_addr=SRAM_BANK_A_OFFSET, name: Optional[str] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fabric = fabric or pytfmbs.Fabric()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.resident = False
        self.tile_mask = (1 << ((out_features + LANES_PER_TILE - 1) // LANES_PER_TILE)) - 1
        self.weight_addr = weight_addr
        self._layer_name = name or f"tfmbs_linear_{id(self)}"

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def prefetch(self, addr=None):
        """Quantizes and moves weights to Fabric memory ahead of time."""
        if self.resident:
            return

        if addr is not None:
            self.weight_addr = addr

        w_ternary = torch.sign(self.weight).to(torch.int8).detach().cpu().numpy()
        tile_data = pack_gemv_weights(w_ternary)

        for t, data in enumerate(tile_data):
            if data is not None:
                # Load to weight bank of tile t
                self.fabric.load(self.weight_addr + t * SRAM_TILE_STRIDE, data)

        self.resident = True

    def load_to_fabric(self):
        """Deprecated alias for prefetch() using default address."""
        self.prefetch()

    def forward(self, input, next_layer=None):
        if not self.resident:
            self.load_to_fabric()

        return TFMBSLinearFunction.apply(
            input, self.fabric, self.weight_addr, self.bias,
            self.in_features, self.out_features, self.tile_mask, next_layer
        )

    @property
    def layer_name(self) -> str:
        return self._layer_name

    @property
    def telemetry_hint(self) -> dict[str, float | int | str]:
        """Telemetry hints (layer name, sparsity, tile mask, fusion metadata)."""
        return {
            "layer": self.layer_name,
            "sparsity": self._compute_sparsity(),
            "tile_mask": int(self.tile_mask),
            "fusion_order": [self.layer_name],
            "fusion_sparsity": self._compute_sparsity(),
            "zero_count": self._zero_count(),
            "total": self._total_elements(),
        }

    def _zero_count(self) -> int:
        weight = self.weight.detach()
        if weight.numel() == 0:
            return 0
        return int(torch.count_nonzero(weight == 0).item())

    def _total_elements(self) -> int:
        return int(self.weight.numel())

    def _compute_sparsity(self) -> float:
        weight = self.weight.detach()
        total = weight.numel()
        if total == 0:
            return 0.0
        zero_count = torch.count_nonzero(weight == 0).item()
        return float(zero_count) / float(total)

class TFMBSSequential(nn.Sequential):
    """
    A Sequential container that automatically handles layer pipelining.
    """
    def forward(self, input):
        for i, module in enumerate(self):
            # Only pass next_layer if it is a TFMBSLinear instance to avoid AttributeErrors
            next_module = self[i+1] if i+1 < len(self) else None
            if not isinstance(next_module, TFMBSLinear):
                next_module = None

            if isinstance(module, TFMBSLinear):
                input = module(input, next_layer=next_module)
            else:
                input = module(input)
        return input

    def telemetry_hint(self) -> dict[str, float | int | str | list[str]]:
        """Aggregate telemetry hints from each TFMBSLinear in the sequence."""
        hints = [module.telemetry_hint for module in self if isinstance(module, TFMBSLinear)]
        if not hints:
            return {}
        fusion_order = []
        total_zero = 0
        total = 0
        tile_masks = []
        for hint in hints:
            fusion_order.extend(hint.get("fusion_order", []))
            total_zero += hint.get("zero_count", 0)
            total += hint.get("total", 0)
            tile_masks.append(hint.get("tile_mask", 0))
        fusion_sparsity = float(total_zero) / float(total) if total else 0.0
        return {
            "layer": "->".join(fusion_order),
            "fusion_order": fusion_order,
            "fusion_sparsity": fusion_sparsity,
            "sparsity": fusion_sparsity,
            "tile_mask": max(tile_masks) if tile_masks else 0,
        }
