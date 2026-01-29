//===- TfmbsOps.td - TFMBS Dialect operations -------------------*- tablegen -*-===//
//
// This file contains a lightweight TableGen-based definition of the operations
// that will appear in the TFMBS dialect. It can be consumed by mlir-tblgen to
// generate the standard dialect header once the build system has integrated
// MLIR TableGen support.
//
include "mlir/IR/Dialect.h"
include "mlir/IR/OpBase.td"

def Tfmbs_Dialect : Dialect {
  let name = "tfmbs";
  let description = "Dialect for the Ternary Fabric semantic operations";
}

def TfmbsPackOp : Op<TfmbsPackOp, [NoSideEffect]> {
  let summary = "Pack a dense tensor into PT-5 format.";
  let description = [{
    Loads a host tensor and writes the quantized PT-5 representation into the
    fabric's local memory.
  }];
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst
  );
}

def TfmbsTransferOp : Op<TfmbsTransferOp> {
  let summary = "Direct copy between fabric memory banks.";
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    AnyType:$len
  );
}

def TfmbsDmaLoadOp : Op<TfmbsDmaLoadOp> {
  let summary = "Issue a DMA load into the fabric from host memory.";
  let arguments = (ins
    AnyType:$host,
    AnyType:$fabric
  );
}

def TfmbsGemvOp : Op<TfmbsGemvOp> {
  let summary = "High-level GEMV kernel targeting the ternary fabric.";
  let arguments = (ins
    AnyType:$weight,
    AnyType:$input,
    AnyType:$output
  );
  let assemblyFormat = "$weight, $input, $output attr-dict";
}
