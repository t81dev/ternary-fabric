#include "TfmbsDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::tfmbs;

#define GET_OP_CLASSES
#include "TfmbsOps.cpp.inc"
#undef GET_OP_CLASSES

TfmbsDialect::TfmbsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TfmbsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "TfmbsOps.cpp.inc"
#undef GET_OP_LIST
      >();
}

void TfmbsDialect::registerDialect(DialectRegistry &registry) {
  registry.insert<TfmbsDialect>();
}

namespace mlir {
namespace tfmbs {

void registerTfmbsDialect(DialectRegistry &registry) {
  TfmbsDialect::registerDialect(registry);
}

} // namespace tfmbs
} // namespace mlir
