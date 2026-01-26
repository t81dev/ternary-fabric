#include <Python.h>
#include "tfmbs.h"
#include <sys/mman.h>
#include <fcntl.h>

// Mock address for the AXI control registers (Physical address on SoC)
#define FABRIC_CTRL_BASE 0x40000000 

typedef struct {
    PyObject_HEAD
    void* mmio_ptr;
} FabricObject;

static int Fabric_init(FabricObject *self, PyObject *args, PyObject *kwds) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not open /dev/mem. Root required.");
        return -1;
    }
    // Map the AXI register space into Python's memory space
    self->mmio_ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, FABRIC_CTRL_BASE);
    close(fd);
    return 0;
}

static PyObject* Fabric_run(FabricObject *self, PyObject *args) {
    unsigned long base_addr;
    int depth, lanes, stride, kernel;

    if (!PyArg_ParseTuple(args, "kiiii", &base_addr, &depth, &lanes, &stride, &kernel))
        return NULL;

    // Direct Register Access (Matches our axi_interconnect_v1.v map)
    volatile uint32_t* regs = (uint32_t*)self->mmio_ptr;
    
    regs[2] = (uint32_t)base_addr; // 0x08: Base Addr
    regs[3] = (uint32_t)depth;     // 0x0C: Depth
    regs[4] = (uint32_t)stride;    // 0x10: Stride
    regs[0] = 0x1;                 // 0x00: Start Bit

    // Poll for completion (Bit 1 of Status register 0x04)
    while(!(regs[1] & 0x2)); 

    Py_RETURN_NONE;
}

// ... (Standard Python Module Boilerplate)