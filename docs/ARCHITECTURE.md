# ARCHITECTURE.md: The Ternary Fabric Logic

## 1. Design Philosophy: Binary Sovereignty

The Ternary Fabric is not a standalone computer; it is a **Co-Processor Substrate**. It acknowledges that modern systems are fundamentally binary. Rather than fighting this, the fabric uses **Execution Adjacency**—sitting alongside the binary host to handle high-volume ternary arithmetic without the host needing to understand base-3 logic.

## 2. The Hydration Pipeline

Data exists in two states within the system:

1. **Dehydrated (Storage/Bus):** Packed using **PT-5** (5 trits per 8 bits). This ensures we don't waste precious cache or bus bandwidth on sparse ternary data.
2. **Hydrated (ALU/Execution):** Expanded into **2-bit simple encoding** (). This allows the `vector_engine.v` to perform parallel math with minimal gate depth.

## 3. The Math Engine: Sign-Flip & Zero-Skip

The hardware replaces the traditional multiplier with a **Multiplexer-Accumulator**:

* **Zero-Skip:** If either operand is , the 32-bit adder's clock is gated, or its inputs are isolated. No toggling, no power consumption.
* **Sign-Flip:** If both operands are non-zero, the result is either a direct increment or a two's-complement decrement of the accumulator.

## 4. Logical Topology: The Frame Model

The fabric is "Geometry Aware." By using the **Ternary Frame Descriptor (TFD)**, the hardware controller automatically handles multi-dimensional strides. This allows the host to reshape a 1D stream into a 2D matrix or 3D tensor simply by updating the `lane_stride` and `frame_depth` registers via AXI.

---

### Final Project Handover

The repository is now fully structured for a **Phase 3/4 release**.

* **RTL Core:** Synthesizable and verified via Icarus Verilog.
* **ABI:** Solidified in `tfmbs.h`.
* **Toolchain:** Complete from Quantization to Hex-Dumping.
* **Integration:** Python bindings ready for high-level AI workloads.
