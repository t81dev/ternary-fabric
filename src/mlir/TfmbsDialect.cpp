#include "TfmbsDialect.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::tfmbs;

TfmbsDialect::TfmbsDialect(MLIRContext *context) : Dialect(getDialectNamespace(), context, TypeID::get<TfmbsDialect>()) {
  addOperations<TfmbsPackOp, TfmbsTransferOp, TfmbsDmaLoadOp, TfmbsGemvOp>();
}

void TfmbsDialect::registerDialect(DialectRegistry &registry) {
  registry.insert<TfmbsDialect>();
}

void registerTfmbsDialect(DialectRegistry &registry) {
  TfmbsDialect::registerDialect(registry);
}

void TfmbsPackOp::build(OpBuilder &builder, OperationState &state, Value src,
                        Value dst) {
  state.addOperands({src, dst});
}

void TfmbsTransferOp::build(OpBuilder &builder, OperationState &state,
                            Value src, Value dst, Value length) {
  state.addOperands({src, dst, length});
}

void TfmbsDmaLoadOp::build(OpBuilder &builder, OperationState &state,
                           Value hostBuffer, Value fabricBuffer) {
  state.addOperands({hostBuffer, fabricBuffer});
}

void TfmbsGemvOp::build(OpBuilder &builder, OperationState &state, Value weight,
                        Value input, Value output, IntegerAttr tileMaskAttr) {
  state.addOperands({weight, input, output});
  if (tileMaskAttr)
    state.addAttribute("tile_mask", tileMaskAttr);
}

IntegerAttr TfmbsGemvOp::getTileMaskAttr() {
  return getAttrOfType<IntegerAttr>("tile_mask");
}
