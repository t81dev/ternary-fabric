## 1. Overview

The **AI Acceleration** specification defines how the ternary fabric performs high-efficiency machine learning operations. Ternary-native AI (using values ) is fundamentally different from binary or floating-point AI because it eliminates the need for complex multipliers.

In this fabric, "multiplication" is reduced to a simple sign-flip or a zero-skip, significantly reducing power consumption and silicon area.

---

## 2. Ternary Multiplication-Accumulation (T-MAC)

The core primitive of the AI engine is the **T-MAC**. Given a weight  and an input , the operation  is governed by the following logic:

|  (Weight) |  (Input) | Operation | Result () |
| --- | --- | --- | --- |
|  |  | Identity |  |
|  |  | Inversion |  |
|  |  | Nullify/Skip |  |

### 2.1 Zero-Skip Optimization

The fabric MUST implement **Zero-Skip** logic. If either the weight or the input is , the accumulator remains unchanged, and the hardware may skip the power-intensive addition cycle. This provides massive energy efficiency for sparse neural networks.

---

## 3. Vectorized Dot Products

AI workloads utilize the `lane` and `stride` definitions from the Frame Model to perform wide vector operations.

* **Row-Major Consumption:** The AI engine reads Weight Frames and Input Frames simultaneously.
* **Accumulation Width:** While inputs are trits, the **Accumulator** MUST be a wider binary integer (typically 16-bit or 32-bit) to prevent overflow during large summations ().

---

## 4. Activation and Quantization

After the T-MAC phase, the fabric provides specialized "Ternary Activation" units to convert binary accumulators back into ternary trits for the next layer.

### 4.1 Thresholding Logic

The standard activation function is a symmetric threshold:

Where  (delta) is a host-defined hyperparameter provided in the `exec_hints` of the TFD.

---

## 5. Execution Hints for AI

The `exec_hints` field in the TFD is utilized to configure the AI Engine:

* **Bit 0-7:** Kernel Selection (e.g., `0x01` = Dense Layer, `0x02` = Convolution).
* **Bit 8-15:** Quantization Threshold ().
* **Bit 16:** Bias Enable (If set, the first lane of the frame is treated as a bias vector).

---

## 6. Supported Kernels

Compliant AI Engines SHOULD implement the following:

1. **T-GEMV:** Ternary General Matrix-Vector multiplication.
2. **T-CONV:** Ternary 2D Convolution (optimized for zero-skipping on padding).
3. **T-POOL:** Max/Min/Avg pooling across ternary lanes.

---