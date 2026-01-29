#include "TfmbsPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/Config/llvm-config.h"

namespace mlir {
namespace tfmbs {
void registerTfmbsDialect(DialectRegistry &registry);
} // namespace tfmbs
} // namespace mlir

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "tfmbs-plugin", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            tfmbs::registerTfmbsDialect(*registry);
            tfmbs::registerTfmbsPasses();
          }};
}
