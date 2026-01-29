#include "TfmbsDialect.h"
#include "TfmbsPasses.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;
  DialectRegistry registry;
  tfmbs::registerTfmbsDialect(registry);
  registry.insert<linalg::LinalgDialect>();
  context.appendDialectRegistry(registry);

  constexpr StringLiteral moduleStr = R"mlir(
module {
  func @test(%w: memref<4x4xf32>, %x: memref<4xf32>, %y: memref<4xf32>) {
    tfmbs.gemv %w, %x, %y {tile_mask = 1}
    return
  }
}
)mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse embedded module\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(tfmbs::createTfmbsToLinalgPass());
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "TfmbsToLinalgPass failed\n";
    return 1;
  }

  std::string printed;
  {
    llvm::raw_string_ostream os(printed);
    module->print(os);
  }

  if (printed.find("linalg.matmul") == std::string::npos) {
    llvm::errs() << "Expected linalg.matmul in transformed module\n";
    return 1;
  }

  llvm::outs() << "TfmbsToLinalgPass unit test passed\n";
  return 0;
}
