# Ternary Fabric Makefile
CC = gcc
CFLAGS = -I./include -Wall -Wextra -O2
SRC_DIR = src
EX_DIR = examples
BIN_DIR = bin

all: directories mediator example

directories:
	mkdir -p $(BIN_DIR)

mediator: $(SRC_DIR)/mediator_mock.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/mediator_mock $(SRC_DIR)/mediator_mock.c

example: $(EX_DIR)/pt5_pack_example.c
	$(CC) $(CFLAGS) -o $(BIN_DIR)/pt5_example $(EX_DIR)/pt5_pack_example.c

clean:
	rm -rf $(BIN_DIR)

.PHONY: all directories clean