module {
  func.func @torch_tfmbs(%w0 : memref<64x48xf32>, %w1 : memref<48x32xf32>, %input : memref<1x64xf32>, %out0 : memref<1x48xf32>, %out1 : memref<1x32xf32>) {
    tfmbs.gemv %w0 : memref<64x48xf32>, %input : memref<1x64xf32>, %out0 : memref<1x48xf32> {tile_mask = 15, telemetry = {layer = "node_linear", sparsity = 0.0000, tile_mask = 15, fusion_order = ["node_linear", "node_linear_1"], fusion_sparsity = 0.0000}}
    tfmbs.gemv %w1 : memref<48x32xf32>, %out0 : memref<1x48xf32>, %out1 : memref<1x32xf32> {tile_mask = 7, telemetry = {layer = "node_linear_1", sparsity = 0.0000, tile_mask = 7, fusion_order = ["node_linear", "node_linear_1"], fusion_sparsity = 0.0000}}
    func.return
  }
}