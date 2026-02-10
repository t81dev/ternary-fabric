module {
  func.func @test_conv_attn(%input: tensor<1x224x224x3xf32>, %filter: tensor<3x3x3x64xf32>, %output: tensor<1x112x112x64xf32>) {
    // Test Conv2D
    tfmbs.conv2d %input : tensor<1x224x224x3xf32>, %filter : tensor<3x3x3x64xf32>, %output : tensor<1x112x112x64xf32> {tile_mask = 1}

    // Test Attention Fusion Placeholder
    %q_w = "tfmbs.const"() {value = dense<1.0> : tensor<64x64xf32>} : () -> tensor<64x64xf32>
    %attn_in = "tfmbs.const"() {value = dense<1.0> : tensor<1x64xf32>} : () -> tensor<1x64xf32>
    %attn_out = "tfmbs.const"() {value = dense<0.0> : tensor<1x64xf32>} : () -> tensor<1x64xf32>

    tfmbs.fused_attn (%q_w, %attn_in, %attn_out) {tile_mask = 3}

    return
  }
}
