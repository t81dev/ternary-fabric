// RUN: %mlir-opt --load-dialect-plugin=%tfmbs-plugin --pass-pipeline=builtin.module(tfmbs-to-linalg) %s | %filecheck %s

module {
  func.func @gemv(%weights: memref<4x4xf32>, %input: memref<4x4xf32>, %output: memref<4x4xf32>) {
    tfmbs.gemv %weights : memref<4x4xf32>, %input : memref<4x4xf32>, %output : memref<4x4xf32> {tile_mask = 1 : i64}
    func.return
  }
}

// CHECK: linalg.matmul
// CHECK-NOT: tfmbs.gemv
