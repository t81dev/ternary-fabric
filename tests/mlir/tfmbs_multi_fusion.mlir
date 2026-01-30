module {
  func.func @multi_fused(%w0 : memref<4x4xf32>, %w1 : memref<4x4xf32>, %w2 : memref<4x4xf32>, %input : memref<4x4xf32>, %mid0 : memref<4x4xf32>, %mid1 : memref<4x4xf32>, %output : memref<4x4xf32>) {
    tfmbs.gemv %w0 : memref<4x4xf32>, %input : memref<4x4xf32>, %mid0 : memref<4x4xf32> {tile_mask = 1, telemetry = {layer = "stage0", sparsity = 0.0000}}
    tfmbs.gemv %w1 : memref<4x4xf32>, %mid0 : memref<4x4xf32>, %mid1 : memref<4x4xf32> {tile_mask = 1, telemetry = {layer = "stage1", sparsity = 0.0000}}
    tfmbs.gemv %w2 : memref<4x4xf32>, %mid1 : memref<4x4xf32>, %output : memref<4x4xf32> {tile_mask = 1, telemetry = {layer = "stage2", sparsity = 0.0000}}
    func.return
  }
}
