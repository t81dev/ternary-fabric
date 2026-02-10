//===- TfmbsOps.td - TFMBS dialect definition -----------------*- tablegen -*-===//
//
// This file defines the Tfmbs dialect and its operations for the Ternary Fabric.
//
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Tfmbs_Dialect : Dialect {
  let name = "tfmbs";
  let description = "Dialect for the Ternary Fabric semantic operations.";
  let cppNamespace = "::mlir::tfmbs";
}

/// Base class for Tfmbs operations so we don't repeat the dialect reference.
class TfmbsOp<string mnemonic> : Op<Tfmbs_Dialect, mnemonic> {
}

def TfmbsPackOp : TfmbsOp<"pack"> {
  let summary = "Pack a dense tensor into PT-5 format.";
  let description = [{
    Loads a host tensor and writes the quantized PT-5 representation into the
    fabric's local memory.
  }];
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst
  );
  let assemblyFormat = "$src `:` type($src) `,` $dst `:` type($dst) attr-dict";
}

def TfmbsTransferOp : TfmbsOp<"transfer"> {
  let summary = "Direct copy between fabric memory banks.";
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    AnyType:$len
  );
  let assemblyFormat = "$src `:` type($src) `,` $dst `:` type($dst) `,` $len `:` type($len) attr-dict";
}

def TfmbsDmaLoadOp : TfmbsOp<"dma_load"> {
  let summary = "Issue a DMA load into the fabric from host memory.";
  let arguments = (ins
    AnyType:$host,
    AnyType:$fabric
  );
  let assemblyFormat = "$host `:` type($host) `,` $fabric `:` type($fabric) attr-dict";
}

def TfmbsGemvOp : TfmbsOp<"gemv"> {
  let summary = "High-level GEMV kernel targeting the ternary fabric.";
  let arguments = (ins
    AnyType:$weight,
    AnyType:$input,
    AnyType:$output,
    DefaultValuedOptionalAttr<I64Attr, "0">:$tile_mask,
    OptionalAttr<DictionaryAttr>:$telemetry
  );
  let assemblyFormat = "$weight `:` type($weight) `,` $input `:` type($input) `,` $output `:` type($output) attr-dict";
}

def TfmbsFusedGemvOp : TfmbsOp<"fused_gemv"> {
  let summary = "Fuse sequential GEMV kernels that share telemetry hints.";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedOptionalAttr<I64Attr, "0">:$tile_mask,
    OptionalAttr<ArrayAttr>:$telemetry
  );
  let description = [{
    Represents a batch of GEMV kernels that share the same tile mask
    and telemetry attributes, allowing the fusion pass to collapse
    several kernels into a single scheduling unit.
    The `operands` are a repeating sequence of weight/input/output for each stage.
  }];
}

def TfmbsConv2DOp : TfmbsOp<"conv2d"> {
  let summary = "2D Convolution kernel targeting the ternary fabric.";
  let arguments = (ins
    AnyType:$input,
    AnyType:$filter,
    AnyType:$output,
    DefaultValuedOptionalAttr<I64Attr, "0">:$tile_mask,
    OptionalAttr<DictionaryAttr>:$telemetry
  );
  let assemblyFormat = "$input `:` type($input) `,` $filter `:` type($filter) `,` $output `:` type($output) attr-dict";
}

def TfmbsFusedAttnOp : TfmbsOp<"fused_attn"> {
  let summary = "Fused Attention block (QKV matmuls + Softmax + Proj).";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedOptionalAttr<I64Attr, "0">:$tile_mask,
    OptionalAttr<ArrayAttr>:$telemetry
  );
  let description = [{
    Represents a fused attention block consisting of Q, K, V matmuls,
    softmax scaling, and output projection.
  }];
}

def TfmbsSoftmaxOp : TfmbsOp<"softmax"> {
  let summary = "Softmax activation targeting the fabric.";
  let arguments = (ins AnyType:$input, AnyType:$output);
  let assemblyFormat = "$input `:` type($input) `,` $output `:` type($output) attr-dict";
}
