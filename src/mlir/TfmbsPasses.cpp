#include "TfmbsDialect.h"
#include "TfmbsPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tfmbs;

namespace {

static int64_t getTileMaskValue(TfmbsGemvOp op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("tile_mask"))
    return attr.getValue().getSExtValue();
  return 0;
}

/// Pass that fuses sequential tfmbs.gemv ops sharing the same hints.
struct TfmbsFusionPass : public PassWrapper<TfmbsFusionPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    OpBuilder builder(function.getContext());
    for (auto op : llvm::make_early_inc_range(function.getOps<TfmbsGemvOp>())) {
      Value intermediate = op.getOutput();
      if (!intermediate.hasOneUse())
        continue;
      auto *user = *intermediate.user_begin();
      auto following = dyn_cast<TfmbsGemvOp>(user);
      if (!following || following.getInput() != intermediate)
        continue;
      int64_t maskA = getTileMaskValue(op);
      int64_t maskB = getTileMaskValue(following);
      if (maskA != maskB)
        continue;

      builder.setInsertionPoint(op);
      SmallVector<Value, 6> fusedOperands = {
          op.getWeight(), op.getInput(), op.getOutput(),
          following.getWeight(), following.getInput(), following.getOutput(),
      };

      SmallVector<Attribute, 2> telemetryRecords;
      bool hasTelemetry = false;
      auto captureTelemetry = [&](TfmbsGemvOp source) {
        if (auto dict = source->getAttrOfType<DictionaryAttr>("telemetry")) {
          telemetryRecords.push_back(dict);
          if (!dict.empty())
            hasTelemetry = true;
        } else {
          telemetryRecords.push_back(builder.getDictionaryAttr({}));
        }
      };
      captureTelemetry(op);
      captureTelemetry(following);

      ArrayAttr telemetryAttr = hasTelemetry ? builder.getArrayAttr(telemetryRecords) : ArrayAttr();
      builder.create<TfmbsFusedGemvOp>(
          op.getLoc(), ValueRange(fusedOperands), builder.getI64IntegerAttr(maskA), telemetryAttr);
      following.erase();
      op.erase();
    }
  }
};

/// Pass that lowers tfmbs.gemv ops (including fused) into `linalg.matmul`.
struct TfmbsToLinalgPass : public PassWrapper<TfmbsToLinalgPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *operation) {
      if (auto fused = dyn_cast<TfmbsFusedGemvOp>(operation)) {
        operation->getContext()->getOrLoadDialect<linalg::LinalgDialect>();
        OpBuilder builder(operation);
        ValueRange operands = fused.getOperands();
        if (operands.size() % 3 != 0)
          return;

        for (unsigned i = 0; i < operands.size() / 3; ++i) {
          Value rhs = operands[3 * i];
          Value lhs = operands[3 * i + 1];
          Value dst = operands[3 * i + 2];
          builder.create<linalg::MatmulOp>(operation->getLoc(), ValueRange{lhs, rhs}, ValueRange{dst});
        }
        operation->erase();
        return;
      }

      if (auto op = dyn_cast<TfmbsGemvOp>(operation)) {
        op.getContext()->getOrLoadDialect<linalg::LinalgDialect>();
        OpBuilder builder(op.getOperation());
        builder.create<linalg::MatmulOp>(op.getLoc(), ValueRange{op.getInput(), op.getWeight()}, ValueRange{op.getOutput()});
        op.erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> tfmbs::createTfmbsToLinalgPass() {
  return std::make_unique<TfmbsToLinalgPass>();
}

std::unique_ptr<Pass> tfmbs::createTfmbsFusionPass() {
  return std::make_unique<TfmbsFusionPass>();
}

namespace {
static PassPipelineRegistration<> pipeline(
    "tfmbs-to-linalg", "Lower tfmbs ops to linalg.matmul",
    [](OpPassManager &pm) {
      pm.nest<func::FuncOp>().addPass(std::make_unique<TfmbsFusionPass>());
      pm.addPass(std::make_unique<TfmbsToLinalgPass>());
    });
} // namespace

void tfmbs::registerTfmbsPasses() {
  (void)pipeline;
}
