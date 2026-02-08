# Phase 3 Integration Handoff Map

Roadmap linkage:
- `t81-roadmap#16`
- `t81-roadmap/PHASE3_MILESTONE_MATRIX.md` (`P3-M2`)
- Implementation tracker: `ternary-fabric#42`

## Integration Checklist

- [x] Integration handoff path from fabric experiments to ecosystem validation documented.
- [x] Measurable interface signals and metrics listed.
- [x] Reproducible evidence command path documented.

## Reproducible Evidence Path

Run:

```bash
python3 tools/reference_integration.py
```

Optional telemetry validation path:

```bash
tools/run_hw_dma_telemetry.sh
python3 tools/adaptive_dashboard.py
```

## Measurable Interface Signals

Primary metrics/signals used for handoff:

1. `zero_skips`
2. `active_ops`
3. `fabric_cost`
4. `residency_hits` / `residency_misses`
5. offload/fallback counters emitted by runtime telemetry paths

Primary references:
- `BENCHMARKS.md`
- `docs/18_WORKLOADS_METRICS.md`
- `docs/FABRIC_ILLUSION_CONTRACT.md`

## Cross-Repo Handoff Targets

1. `t81-benchmarks`: consume comparable metrics in benchmark/publication artifacts.
2. `t81-roadmap`: track milestone progress and attach evidence links.
3. `t81-hardware`: align telemetry/signal semantics for hardware-facing checkpoints.
