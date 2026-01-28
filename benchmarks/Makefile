CC=gcc
CFLAGS=-O3 -I../include -I../src
LDFLAGS=-L../bin -ltfmbs_device -lpthread

# Reorganized Phase 18 Structure
SYNTHETIC_BINS=synthetic/saturation synthetic/density
GEMM_BINS=gemm/gemm_bench
LSTM_BINS=lstm/lstm_bench
TEMPORAL_BINS=temporal/lstm_persistent

ALL_BINS=$(SYNTHETIC_BINS) $(GEMM_BINS) $(LSTM_BINS) $(TEMPORAL_BINS)

all: $(ALL_BINS)

synthetic/%: synthetic/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

gemm/%: gemm/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

lstm/%: lstm/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

temporal/%: temporal/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bench: all
	@echo "Running Phase 18 Benchmarks..."
	@for bin in $(ALL_BINS); do \
		echo "Running $$bin..."; \
		LD_LIBRARY_PATH=../bin ./$$bin; \
	done

clean:
	rm -f $(ALL_BINS)
