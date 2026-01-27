# Ternary Fabric Visualizations

This document provides visual representations of the Ternary Fabric architecture and its operational workflows.

## 1. Full-Stack Architecture

The following diagram illustrates the relationship between the Host CPU, the Frame Controller, and the multi-tile Ternary Fabric.

```mermaid
graph TD
    subgraph Binary Host
        CPU[Host CPU - x86/ARM]
        App[AI Application/Runtime]
        Driver[pytfmbs Driver]
    end

    subgraph Ternary Fabric Core
        FC[Frame Controller]
        AXI[AXI4-Lite / AXI-Stream]

        subgraph Tile 0
            T0_SRAM_W[Weight SRAM]
            T0_SRAM_I[Input SRAM]
            T0_VE[Vector Engine]
            T0_Results[Result Registers]
        end

        subgraph Tile 1
            T1_SRAM_W[Weight SRAM]
            T1_SRAM_I[Input SRAM]
            T1_VE[Vector Engine]
            T1_Results[Result Registers]
        end

        subgraph "..."
        end

        subgraph Tile N
            TN_SRAM_W[Weight SRAM]
            TN_SRAM_I[Input SRAM]
            TN_VE[Vector Engine]
            TN_Results[Result Registers]
        end
    end

    App --> Driver
    Driver --> AXI
    AXI --> FC
    FC --> T0_VE
    FC --> T1_VE
    FC --> TN_VE

    T0_SRAM_W --> T0_VE
    T0_SRAM_I --> T0_VE
    T1_SRAM_W --> T1_VE
    T1_SRAM_I --> T1_VE

    T0_VE --> T0_Results
    T1_VE --> T1_Results
```

## 2. Hydration & SIMD Execution Workflow

The process of moving from packed PT-5 data to SIMD results is shown below.

```mermaid
sequenceDiagram
    participant Host as Binary Host
    participant DMA as AXI DMA / SRAM
    participant Unpacker as PT-5 Unpacker
    participant Lane as TPE Lane (SIMD)
    participant Acc as Accumulator

    Host->>DMA: Load Packed PT-5 Data (Weights/Inputs)
    Host->>Host: Create TFD (Ternary Frame Descriptor)
    Host->>DMA: Write TFD to Frame Controller

    Note over DMA, Acc: Execution Loop (per Trit)

    DMA->>Unpacker: Fetch 32-bit Packed Word
    Unpacker->>Unpacker: Hydrate trits (5 per byte)
    Unpacker->>Lane: Parallel Dispatch (15 Lanes)

    Note over Lane: Zero-Skip Check
    alt Trit is 0
        Lane->>Lane: Gated (No toggle)
    else Trit is +1 or -1
        Lane->>Acc: Apply Weight/Input Logic
        Acc->>Acc: Accumulate Result
    end

    Note over Host, Acc: End of Frame
    Acc->>Host: Readback Results via AXI
```

## 3. PT-5 Memory Efficiency

PT-5 packing achieves higher density than standard 2-bit-per-trit encoding.

| Format | Trits per Byte | Efficiency |
| :--- | :--- | :--- |
| **Binary (2b/trit)** | 4 | 80.0% (4/5) |
| **PT-5** | 5 | **95.1%** |

*Efficiency calculation based on $\log_2(3^5) / 8$.*
