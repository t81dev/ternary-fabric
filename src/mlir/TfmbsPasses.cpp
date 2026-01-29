#include "TfmbsDialect.h"
#include "TfmbsPasses.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistration.h"

using namespace mlir;
using namespace mlir::tfmbs;

namespace {

/// Pass that lowers tfmbs.gemv ops into `linalg.matmul`.
struct TfmbsToLinalgPass : public PassWrapper<TfmbsToLinalgPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](TfmbsGemvOp op) {
      OpBuilder builder(op);
      Value lhs = op.getInput();
      Value rhs = op.getWeight();
      Value dst = op.getOutput();
      builder.create<linalg::MatmulOp>(op.getLoc(), ValueRange{lhs, rhs}, ValueRange{dst});
      op.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> tfmbs::createTfmbsToLinalgPass() {
  return std::make_unique<TfmbsToLinalgPass>();
}

void tfmbs::registerTfmbsPasses() {
  PassRegistration<TfmbsToLinalgPass>("tfmbs-to-linalg", "Lower tfmbs ops to linalg.matmul");
}
