#ifndef TFMBS_DIALECT
#define TFMBS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TFMBS Dialect definition.
//===----------------------------------------------------------------------===//

def Tfmbs_Dialect : Dialect {
    let name = "tfmbs";
    let summary = "A dialect for Ternary Fabric Multi-node Bus System (TFMBS).";
    let description = [{
        The TFMBS dialect is designed to represent and optimize operations for
        ternary-native hardware accelerators. It provides first-class support
        for ternary packing (PT-5), zero-skip GEMM/GEMV, and multi-node
        fabric orchestration.
    }];
    let cppNamespace = "::mlir::tfmbs";
}

//===----------------------------------------------------------------------===//
// TFMBS Type definitions.
//===----------------------------------------------------------------------===//

def Tfmbs_TernaryType : TypeDef<Tfmbs_Dialect, "Ternary"> {
    let mnemonic = "ternary";
    let summary = "A ternary element (-1, 0, +1).";
    let description = [{
        Represents a single ternary digit (trit). Physical representation
        is usually 2 bits in binary memory or packed via PT-5.
    }];
}

#endif // TFMBS_DIALECT
