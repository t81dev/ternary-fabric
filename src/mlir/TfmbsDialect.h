#ifndef TFMBS_DIALECT_H
#define TFMBS_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include <cstdint>

namespace mlir {
class DialectRegistry;
class MLIRContext;
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

void registerTfmbsDialect(DialectRegistry &registry);

} // namespace tfmbs
} // namespace mlir

#define GET_OP_CLASSES
#include "TfmbsOps.h.inc"
#undef GET_OP_CLASSES

#endif // TFMBS_DIALECT_H
