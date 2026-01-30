#include "TfmbsDialect.h"
#include "TfmbsPasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::tfmbs;

namespace {

/// Pass that lowers tfmbs.gemv ops into `linalg.matmul`.
struct TfmbsToLinalgPass : public PassWrapper<TfmbsToLinalgPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](TfmbsGemvOp op) {
      op.getContext()->getOrLoadDialect<linalg::LinalgDialect>();
      OpBuilder builder(op.getOperation());
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

namespace {
static PassPipelineRegistration<> pipeline(
    "tfmbs-to-linalg", "Lower tfmbs ops to linalg.matmul",
    [](OpPassManager &pm) { pm.addPass(std::make_unique<TfmbsToLinalgPass>()); });
} // namespace

void tfmbs::registerTfmbsPasses() {
  (void)pipeline;
}
