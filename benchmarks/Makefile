CC=gcc
CFLAGS=-O3 -I../include -I../src
LDFLAGS=-L../bin -ltfmbs_device -lpthread

TARGETS=layer1_synthetic layer2_kernels

all: $(TARGETS)

layer1_synthetic: layer1_synthetic.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

layer2_kernels: layer2_kernels.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
