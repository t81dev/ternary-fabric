# 🧭 Ternary-Fabric × llama.cpp Acceleration Roadmap

---

## Phase 0 — Positioning & Invariants

Before touching code, lock these invariants:

* llama.cpp remains **orchestrator + sampler**.
* Ternary-Fabric is a **projection / GEMV accelerator**.
* Only accelerate:

  * matmul / gemv
  * attention projections
  * feed-forward layers
* Never accelerate:

  * tokenizer
  * sampling
  * softmax control
  * KV cache logic

Define success metric early:

```
tokens/sec per watt
bytes moved per token
fabric_time / host_time
```

Not FLOPs.

Deliverable:

* Architecture doc: `TFMBS_LLAMA_INTEGRATION.md`.

---

## Phase 1 — Minimal Backend Hook

Goal: prove Fabric can sit inside llama.cpp without changing quant formats.

### Tasks

1. **Add backend flag**

   ```
   --backend tfmbs
   ```

2. **Create stub backend**

   ```
   ggml_backend_tfmbs.c
   ```

3. **Intercept GEMV**
   Replace only:

   * `ggml_mul_mat`
   * `ggml_vec_dot`

   With:

   ```c
   if (use_tfmbs && supported_type(tensor)) {
       tfmbs_dispatch(...);
   }
   ```

4. **No acceleration yet**

   * Just route calls.
   * Verify correctness.

Deliverable:

* llama.cpp builds with `TFMBS_BACKEND`.
* All tests pass with Fabric stub.

---

## Phase 2 — Fabric Host API Stabilization

Goal: define how llama.cpp talks to Ternary-Fabric.

### Tasks

Define C ABI:

```c
tfmbs_init();
tfmbs_upload_frame();
tfmbs_run_gemv();
tfmbs_wait();
tfmbs_shutdown();
```

Add:

* frame metadata
* tensor stride handling
* async submission option

Establish memory model:

* weights → resident in Fabric
* activations → streamed

Deliverable:

* `include/tfmbs_host.h`
* working simulator path (even if fake at first).

---

## Phase 3 — Quant-Agnostic Acceleration

Goal: accelerate existing quants without inventing new ones.

Instead of defining Q2_T, use **ternary micro-decomposition**.

### Tasks

1. Support these types first:

   * Q4_K
   * Q6_K
   * Q8_0

2. Decompose weights:

```
W = Σ Pi * αi
Pi ∈ {-1,0,+1}
```

Balanced ternary planes.

3. Cache ternary planes per layer on Fabric:

```c
tfmbs_upload_planes(layer_id, planes);
```

4. Execute:

```
for plane i:
    tfmbs_run(Pi, x, tmp)
    y += αi * tmp
```

Host handles accumulation + scale.

5. Enable **zero-skip telemetry**.

Deliverable:

* First real acceleration on projection layers.
* No GGUF changes needed.

---

## Phase 4 — Residency & Reuse

Goal: eliminate redundant movement.

### Tasks

* Upload all layer weights once at model load.
* Keep frames resident in Fabric memory.
* Only move:

  * input activation
  * output vector

Add:

* layer → frame map
* eviction policy for large models

Deliverable:

* Weight bandwidth drops dramatically.
* Fabric becomes a memory-centric accelerator.

---

## Phase 5 — Attention & FFN Coverage

Goal: cover most runtime cost.

Target kernels:

| Layer                 | Offload |
| --------------------- | ------- |
| Q, K, V projections   | ✅       |
| Attention output proj | ✅       |
| FFN up / down         | ✅       |
| Embedding lookup      | ✅       |

Leave on host:

* softmax
* KV cache ops
* sampling

Add pipelining:

```
submit Q,K,V → overlap → wait → host softmax
```

Deliverable:

* Majority of token time runs through Fabric.

---

## Phase 6 — SIMD, Zero-Skip, Broadcast

Goal: activate Fabric’s native advantages.

### Tasks

* Enable SIMD broadcast for activations.
* Exploit:

  * skip when ternary digit = 0
  * compact PT-5 frames
* Track:

  ```
  skip_rate
  lanes_used
  effective_ops
  ```

Tune quant decomposition to maximize skip density without breaking accuracy.

Deliverable:

* Real power + bandwidth advantage, not just compute offload.

---

## Phase 7 — Accuracy Controls & Hybrid Mode

Goal: prevent ternary from hurting quality.

Add policies:

```
--tfmbs-profile fast
--tfmbs-profile balanced
--tfmbs-profile accurate
```

Logic:

```c
if (error_estimate > threshold)
    fallback_to_native();
```

Support mixed layers:

* early layers → Fabric
* late layers → CPU

Deliverable:

* Stable quality with configurable performance.

---

## Phase 8 — Telemetry & Benchmarking

Goal: make performance undeniable.

Add metrics:

* tokens/sec
* bytes/token
* fabric_time vs host_time
* skip_rate
* energy proxy

Benchmark:

* tiny model
* 7B
* sparse vs dense

Produce plots:

```
Baseline llama.cpp vs TFMBs backend
```

Deliverable:

* Reproducible benchmark suite.

---

## Phase 9 — Batch & Pipeline Acceleration

Goal: go beyond single-token loop.

Add:

* multi-token batching
* prefetch frames
* async submission

Pattern:

```
submit token N+1 while host works on token N
```

Deliverable:

* Higher utilization of Fabric.

---

## Phase 10 — Public Integration Layer

Goal: make this usable by others.

Add:

* documentation
* example command:

  ```
  llama-cli --backend tfmbs --profile balanced
  ```
* model compatibility notes

Deliverable:

* Clean user-facing integration.

---

# 🧠 Strategic Milestones

| Phase | Value              |
| ----- | ------------------ |
| 1     | Plumbing           |
| 3     | First acceleration |
| 4     | Real bandwidth win |
| 5     | Token speedup      |
| 6     | Energy advantage   |
| 7     | Quality stability  |
| 8     | Proof              |
| 10    | Adoption           |

---

# 🔑 Core Design Principle

Every phase enforces the same rule:

> **Ternary-Fabric accelerates inner products, not intelligence.**

It moves data cheaper, skips zeros, replaces multiplies with sign logic, and keeps llama.cpp sovereign over control.

That’s exactly aligned with your Duotronic → Fabric pivot.

---
