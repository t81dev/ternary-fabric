#ifndef TFMBS_DIALECT_H
#define TFMBS_DIALECT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"

#include <cstdint>

namespace mlir {
class DialectRegistry;
class MLIRContext;
class Operation;
class OpBuilder;
class OperationState;
class IntegerAttr;
class Type;
} // namespace mlir

namespace mlir {
namespace tfmbs {

/// Identifiers for the supported kernels and dispatch hints. These values
/// mirror the `TFMBS_HINT_KERNEL_MASK` bits referenced in the runtime driver.
enum class TfmbsKernelHint : uint32_t {
  Gemv = 0,
  Tgemm = 1,
  Transfer = 2,
  Unknown = 0xFFFFFFFF
};

class TfmbsDialect : public Dialect {
public:
  explicit TfmbsDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfmbs"; }
  static void registerDialect(DialectRegistry &registry);
};

class TfmbsPackOp
    : public Op<TfmbsPackOp, OpTrait::NOperands<2>, OpTrait::ZeroResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "tfmbs.pack"; }
  static void build(OpBuilder &builder, OperationState &state, Value src,
                    Value dst);

  Value getSource() { return getOperand(0); }
  Value getDestination() { return getOperand(1); }
};

class TfmbsTransferOp
    : public Op<TfmbsTransferOp, OpTrait::NOperands<3>, OpTrait::ZeroResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "tfmbs.transfer"; }
  static void build(OpBuilder &builder, OperationState &state, Value src,
                    Value dst, Value length);

  Value getSource() { return getOperand(0); }
  Value getDestination() { return getOperand(1); }
  Value getLength() { return getOperand(2); }
};

class TfmbsDmaLoadOp
    : public Op<TfmbsDmaLoadOp, OpTrait::NOperands<2>, OpTrait::ZeroResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "tfmbs.dma_load"; }
  static void build(OpBuilder &builder, OperationState &state, Value hostBuffer,
                    Value fabricBuffer);

  Value getHostBuffer() { return getOperand(0); }
  Value getFabricBuffer() { return getOperand(1); }
};

class TfmbsGemvOp
    : public Op<TfmbsGemvOp, OpTrait::NOperands<3>, OpTrait::ZeroResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "tfmbs.gemv"; }
  static void build(OpBuilder &builder, OperationState &state, Value weight,
                    Value input, Value output, IntegerAttr tileMaskAttr);

  Value getWeight() { return getOperand(0); }
  Value getInput() { return getOperand(1); }
  Value getOutput() { return getOperand(2); }
  IntegerAttr getTileMaskAttr();
};

void registerTfmbsDialect(DialectRegistry &registry);

} // namespace tfmbs
} // namespace mlir

#endif // TFMBS_DIALECT_H
