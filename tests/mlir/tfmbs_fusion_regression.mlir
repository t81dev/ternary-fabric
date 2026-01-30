; RUN: %python %S/run_tfmbs_to_linalg.py --mlir=%S/tfmbs_fusion.mlir
; This regression ensures the TfmbsFusionPass still lowers to linalg.matmul with telemetry metadata set.
