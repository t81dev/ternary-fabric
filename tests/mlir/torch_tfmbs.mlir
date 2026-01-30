module {
  func.func @torch_tfmbs(%w0 : memref<12x16xf32>, %w1 : memref<16x12xf32>, %input : memref<2x16xf32>, %out0 : memref<2x12xf32>, %out1 : memref<2x16xf32>) {
    tfmbs.gemv %w0, %input, %out0 {tile_mask = 1, telemetry = {layer = "node_linear", sparsity = 0.0000, tile_mask = 1}}
    tfmbs.gemv %w1, %out0, %out1 {tile_mask = 3, telemetry = {layer = "node_linear_1", sparsity = 0.0000, tile_mask = 3}}
    func.return
  }
}
