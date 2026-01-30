#include "TfmbsDialect.h"
#include "TfmbsPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<linalg::LinalgDialect>();
  mlir::tfmbs::registerTfmbsDialect(registry);
  MLIRContext context(registry);
  context.loadDialect<mlir::tfmbs::TfmbsDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<linalg::LinalgDialect>();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  auto memref4x4 = MemRefType::get({4, 4}, builder.getF32Type());
  auto funcType = builder.getFunctionType({memref4x4, memref4x4, memref4x4}, {});
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test", funcType);
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  builder.create<mlir::tfmbs::TfmbsGemvOp>(builder.getUnknownLoc(),
      entryBlock->getArgument(0), entryBlock->getArgument(1),
      entryBlock->getArgument(2), static_cast<uint64_t>(0), nullptr);
  builder.create<func::ReturnOp>(builder.getUnknownLoc());

  PassManager pm(&context);
  pm.addPass(mlir::tfmbs::createTfmbsToLinalgPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "TfmbsToLinalgPass failed\n";
    return 1;
  }

  std::string printed;
  {
    llvm::raw_string_ostream os(printed);
    module.print(os);
  }

  if (printed.find("linalg.matmul") == std::string::npos) {
    llvm::errs() << "Expected linalg.matmul in transformed module\n";
    return 1;
  }

  llvm::outs() << "TfmbsToLinalgPass unit test passed\n";
  return 0;
}
