#ifndef TFMBS_OPS
#define TFMBS_OPS

include "TfmbsDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TFMBS Op definitions.
//===----------------------------------------------------------------------===//

class Tfmbs_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tfmbs_Dialect, mnemonic, traits>;

def Tfmbs_PackOp : Tfmbs_Op<"pack", [Pure]> {
    let summary = "Pack a dense tensor into ternary format.";
    let description = [{
        Converts a dense tensor (e.g., f32 or i8) into a ternary tensor
        using the PT-5 packing format or raw ternary.
    }];

    let arguments = (ins AnyTensor:$input,
                         DefaultValuedAttr<I32Attr, "1">:$stride,
                         DefaultValuedAttr<F32Attr, "1.0">:$density);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Tfmbs_UnpackOp : Tfmbs_Op<"unpack", [Pure]> {
    let summary = "Unpack a ternary tensor into dense format.";
    let description = [{
        Expands a ternary tensor back into a higher-precision dense tensor.
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Tfmbs_GemvOp : Tfmbs_Op<"gemv", [Pure]> {
    let summary = "Ternary Matrix-Vector Multiplication.";
    let description = [{
        Performs y = Wx where W is a ternary matrix and x is a ternary vector.
        Supports zero-skipping and configurable accumulation precision.
    }];

    let arguments = (ins AnyTensor:$weight,
                         AnyTensor:$input,
                         DefaultValuedAttr<BoolAttr, "true">:$zero_skip);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = "$weight `,` $input attr-dict `:` type($weight) `,` type($input) `->` type($output)";
}

def Tfmbs_TransferOp : Tfmbs_Op<"transfer"> {
    let summary = "Explicit memory transfer between nodes or device memory.";
    let description = [{
        Moves data between host, device, or remote fabric nodes.
    }];

    let arguments = (ins AnyTensor:$input,
                         I32Attr:$src_node,
                         I32Attr:$dst_node,
                         DefaultValuedAttr<BoolAttr, "false">:$is_async);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = "$input `from` $src_node `to` $dst_node attr-dict `:` type($input) `->` type($output)";
}

#endif // TFMBS_OPS
