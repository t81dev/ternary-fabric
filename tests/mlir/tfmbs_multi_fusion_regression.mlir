; RUN: %python %S/run_tfmbs_to_linalg.py --mlir=%S/tfmbs_multi_fusion.mlir
; Ensures multi-stage fusion chains still lower to linalg.matmul post-pass.
