#!/usr/bin/env python3
"""Export a tiny Torch/ONNX model and emit a tfmbs-based MLIR module with telemetry."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

LANES_PER_TILE = 15


def parse_shape(value_proto) -> list[int | None]:
    dims = []
    tensor = value_proto.type.tensor_type
    for dim in tensor.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return dims


def memref_type(dims: list[int | None]) -> str:
    if not dims:
        return "memref<xf32>"
    parts = []
    for dim in dims:
        parts.append(str(dim) if dim and dim > 0 else "?")
    return f"memref<{"x".join(parts)}xf32>"


def build_model(in_features: int, hidden: int, out_features: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_features),
    )


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, input_shape: tuple[int, ...]):
    model.eval()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        verbose=False,
        do_constant_folding=True,
    )


def collect_matmuls(model_path: Path) -> tuple[list[dict], list[int | None]]:
    graph = onnx.load(model_path).graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    input_dims = {value.name: parse_shape(value) for value in graph.input}

    matmul_infos = []
    for idx, node in enumerate(graph.node):
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        weights = initializers.get(node.input[1])
        if weights is None:
            continue
        out_features, in_features = weights.shape
        batch = input_dims.get(node.input[0], [None])[0]
        zero_count = int(np.count_nonzero(weights == 0))
        sparsity = float(zero_count) / weights.size
        tile_count = math.ceil(out_features / LANES_PER_TILE) if out_features else 0
        tile_mask = (1 << tile_count) - 1 if tile_count else 0
        matmul_infos.append({
            "label": node.name or f"matmul_{idx}",
            "in_features": int(in_features),
            "out_features": int(out_features),
            "batch_dim": batch,
            "sparsity": sparsity,
            "tile_mask": tile_mask,
            "zero_count": zero_count,
            "total": int(weights.size),
        })
    return matmul_infos, input_dims.get("input", [])


def build_mlir(matmuls: list[dict], input_dims: list[int | None]) -> str:
    lines: list[str] = ["module {"]
    args: list[tuple[str, str]] = []

    for i, info in enumerate(matmuls):
        args.append((f"%w{i}", memref_type([info["in_features"], info["out_features"]])))

    input_type = memref_type(input_dims)
    args.append(("%input", input_type))

    for i, info in enumerate(matmuls):
        dims = [input_dims[0] if input_dims else None, info["out_features"]]
        args.append((f"%out{i}", memref_type(dims)))

    signature = ", ".join(f"{name} : {typ}" for name, typ in args)
    lines.append(f"  func.func @torch_tfmbs({signature}) {{")

    prev_input = "%input"
    arg_types = {name: typ for name, typ in args}
    fusion_order = [info["label"] for info in matmuls]
    total_zero = sum(info["zero_count"] for info in matmuls)
    total_elements = sum(info["total"] for info in matmuls)
    fusion_sparsity = float(total_zero) / float(total_elements) if total_elements else 0.0
    fusion_order_str = ", ".join(f"\"{name}\"" for name in fusion_order)
    for idx, info in enumerate(matmuls):
        telemetry = (
            f"{{layer = \"{info['label']}\", sparsity = {info['sparsity']:.4f}, "
            f"tile_mask = {info['tile_mask']}, fusion_order = [{fusion_order_str}], fusion_sparsity = {fusion_sparsity:.4f}}}"
        )
        lines.append(
            f"    tfmbs.gemv %w{idx} : {arg_types[f'%w{idx}']}, "
            f"{prev_input} : {arg_types[prev_input]}, "
            f"%out{idx} : {arg_types[f'%out{idx}']} {{tile_mask = {info['tile_mask']}, telemetry = {telemetry}}}"
        )
        prev_input = f"%out{idx}"

    lines.append("    func.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Torch/ONNX to TFMBS MLIR helper")
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("tests/mlir/torch_tfmbs.onnx"),
        help="Path to the exported ONNX model",
    )
    parser.add_argument(
        "--emit",
        type=Path,
        default=Path("tests/mlir/torch_tfmbs.mlir"),
        help="Destination MLIR file",
    )
    args = parser.parse_args()

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    args.emit.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.in_features, args.hidden, args.out_features)
    export_to_onnx(model, args.onnx, (args.batch, args.in_features))

    matmuls, input_dims = collect_matmuls(args.onnx)
    if not matmuls:
        raise SystemExit("No MatMul nodes found in ONNX graph.")

    mlir = build_mlir(matmuls, input_dims)
    args.emit.write_text(mlir)
    print(f"Emitted tfmbs MLIR with {len(matmuls)} GEMV(s) to {args.emit}")
    print(mlir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
