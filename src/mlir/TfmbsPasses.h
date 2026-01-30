#ifndef TFMBS_PASSES_H
#define TFMBS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
}

namespace mlir {
namespace tfmbs {

std::unique_ptr<::mlir::Pass> createTfmbsToLinalgPass();
std::unique_ptr<::mlir::Pass> createTfmbsFusionPass();

void registerTfmbsPasses();

} // namespace tfmbs
} // namespace mlir

#endif // TFMBS_PASSES_H
