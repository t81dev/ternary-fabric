#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tfmbs_device.h"

int main() {
    printf("🧪 Testing Multi-Tile Scaling (Phase 11)...\n");

    int rows = 64;
    int cols = 64;

    int8_t* h_weights = malloc(rows * cols);
    int8_t* h_input = malloc(cols);
    int32_t* h_output = malloc(rows * sizeof(int32_t));

    for (int i = 0; i < rows * cols; i++) h_weights[i] = (i % 3) - 1;
    for (int i = 0; i < cols; i++) h_input[i] = (i % 2) ? 1 : -1;

    void* d_weights = fabric_alloc(rows * cols);
    void* d_input = fabric_alloc(cols);
    void* d_output = fabric_alloc(rows * sizeof(int32_t));

    fabric_memcpy_to(d_weights, h_weights, rows * cols, 1);
    fabric_memcpy_to(d_input, h_input, cols, 1);

    // Test with 1 tile
    printf("\n--- Testing with 1 Tile (Mask 0x01) ---\n");
    setenv("FABRIC_TILE_MASK", "0x01", 1);
    fabric_handle_t h1 = fabric_exec_gemv_async(d_weights, d_input, d_output, rows, cols);
    fabric_wait(h1);

    fabric_metrics_t m1;
    fabric_get_metrics(&m1);
    printf("Metrics (1 Tile): Lanes used = %d\n", m1.lanes_used);
    assert(m1.lanes_used == 15);

    // Test with 4 tiles
    printf("\n--- Testing with 4 Tiles (Mask 0x0F) ---\n");
    setenv("FABRIC_TILE_MASK", "0x0F", 1);
    fabric_handle_t h4 = fabric_exec_gemv_async(d_weights, d_input, d_output, rows, cols);
    fabric_wait(h4);

    fabric_metrics_t m4;
    fabric_get_metrics(&m4);
    printf("Metrics (4 Tiles): Lanes used = %d\n", m4.lanes_used);
    assert(m4.lanes_used == 60);

    // Test with 2 tiles
    printf("\n--- Testing with 2 Tiles (Mask 0x03) ---\n");
    setenv("FABRIC_TILE_MASK", "0x03", 1);
    fabric_handle_t h2 = fabric_exec_gemv_async(d_weights, d_input, d_output, rows, cols);
    fabric_wait(h2);

    fabric_metrics_t m2;
    fabric_get_metrics(&m2);
    printf("Metrics (2 Tiles): Lanes used = %d\n", m2.lanes_used);
    assert(m2.lanes_used == 30);

    fabric_free(d_weights);
    fabric_free(d_input);
    fabric_free(d_output);
    free(h_weights);
    free(h_input);
    free(h_output);

    printf("\n⭐ Multi-Tile Scaling Validation PASSED\n");
    return 0;
}
