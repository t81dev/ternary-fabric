// RUN: mlir-opt --pass-pipeline=tfmbs-to-linalg %s | FileCheck %s

module {
  func.func @gemv(%weights: memref<4x4xf32>, %input: memref<4x4xf32>, %output: memref<4x4xf32>) {
    tfmbs.gemv %weights, %input, %output {tile_mask = 1}
    func.return
  }
}

// CHECK: linalg.matmul
// CHECK-NOT: tfmbs.gemv
