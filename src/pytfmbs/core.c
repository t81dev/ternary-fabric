#include <Python.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h> // Required for calloc/free
#include "tfmbs.h"

typedef struct {
    PyObject_HEAD
    void* mmio_ptr;
} FabricObject;

// 1. Cleanup handler for the Python Object
static void Fabric_dealloc(FabricObject *self) {
    if (self->mmio_ptr) {
#if defined(__APPLE__) || defined(MOCK_MODE)
        free(self->mmio_ptr); // Free mock memory
#else
        munmap(self->mmio_ptr, 65536); // Unmap real hardware
#endif
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// 2. Logic Implementations
static int Fabric_init(FabricObject *self, PyObject *args, PyObject *kwds) {
#if defined(__APPLE__) || defined(MOCK_MODE)
    // Mock Mode: Allocate 64KB of RAM to act as registers and SRAM
    printf("[TFMBS] Initializing Mock Mode (Virtual Registers + SRAM)\n");
    self->mmio_ptr = calloc(1, 65536);
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate mock MMIO buffer");
        return -1;
    }
    // Set 'Done' bit in mock status register so polling doesn't hang forever
    // 0x04 is Status. Bit 1 is Done.
    ((uint32_t*)self->mmio_ptr)[1] = 0x2; 
#else
    // Linux/SoC Mode: Real physical memory access
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        // Fallback to mock mode if /dev/mem is not accessible (e.g. in sandbox)
        printf("[TFMBS] /dev/mem not accessible. Falling back to Mock Mode.\n");
        self->mmio_ptr = calloc(1, 65536);
        if (!self->mmio_ptr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate mock MMIO buffer");
            return -1;
        }
        ((uint32_t*)self->mmio_ptr)[1] = 0x2;
        return 0;
    }
    self->mmio_ptr = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x40000000);
    close(fd);
    if (self->mmio_ptr == MAP_FAILED) {
        PyErr_SetString(PyExc_RuntimeError, "mmap failed.");
        return -1;
    }
#endif
    return 0;
}

static PyObject* Fabric_load(FabricObject *self, PyObject *args) {
    uint32_t offset;
    Py_buffer buffer;

    if (!PyArg_ParseTuple(args, "Iy*", &offset, &buffer))
        return NULL;

    if (!self->mmio_ptr) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    if (offset + buffer.len > 65536) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, "Write out of bounds");
        return NULL;
    }

    // In a real hardware scenario with AXI-Lite, we must write 32-bit words.
    // However, the PT-5 data is byte-packed. Our current hardware loader
    // expects one 32-bit AXI write per SRAM word.
    // Let's assume the buffer contains 32-bit words (or we convert them).
    // For simplicity in this mock/loader, we'll write 4 bytes at a time.

    volatile uint32_t* target = (uint32_t*)((uint8_t*)self->mmio_ptr + offset);
    uint32_t* source = (uint32_t*)buffer.buf;
    size_t count = buffer.len / 4;

    for (size_t i = 0; i < count; i++) {
        target[i] = source[i];
    }

#ifdef __APPLE__
    printf("[TFMBS-Mock] Loaded %zu words into 0x%x\n", count, offset);
#endif

    PyBuffer_Release(&buffer);
    Py_RETURN_NONE;
}

static PyObject* Fabric_results(FabricObject *self, PyObject *args) {
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    PyObject* list = PyList_New(15);
    volatile uint32_t* regs = (uint32_t*)self->mmio_ptr;
    for (int i = 0; i < 15; i++) {
        // Results start at offset 0x100, which is index 64 in uint32_t array
        PyList_SetItem(list, i, PyLong_FromLong((int32_t)regs[64 + i]));
    }
    return list;
}

static PyObject* Fabric_run(FabricObject *self, PyObject *args) {
    unsigned long base_addr;
    int depth, lanes, stride, kernel;

    if (!PyArg_ParseTuple(args, "kiiii", &base_addr, &depth, &lanes, &stride, &kernel))
        return NULL;

    volatile uint32_t* regs = (uint32_t*)self->mmio_ptr;
    if (!regs) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    // Write to Registers
    regs[2] = (uint32_t)base_addr; 
    regs[3] = (uint32_t)depth;     
    regs[4] = (uint32_t)stride;    
    
#ifdef __APPLE__
    printf("[TFMBS-Mock] Frame started: Addr=0x%lx, Depth=%d, Kernel=%d\n", base_addr, depth, kernel);
#endif

    regs[0] = 0x1;                 // Start signal

    // On macOS, the mock 'done' bit is pre-set in init to avoid hanging.
    // In hardware, we wait for the RTL to toggle bit 1 of regs[1].
    while(!(regs[1] & 0x2));       

    Py_RETURN_NONE;
}

static PyMethodDef Fabric_methods[] = {
    {"run", (PyCFunction)Fabric_run, METH_VARARGS, "Execute a Ternary Frame Descriptor"},
    {"load", (PyCFunction)Fabric_load, METH_VARARGS, "Load binary data into Fabric SRAM"},
    {"results", (PyCFunction)Fabric_results, METH_NOARGS, "Read accumulated results from the Fabric"},
    {NULL, NULL, 0, NULL} 
};

static PyTypeObject FabricType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pytfmbs.Fabric",
    .tp_doc = "Ternary Fabric Hardware Interface",
    .tp_basicsize = sizeof(FabricObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Fabric_init,
    .tp_dealloc = (destructor)Fabric_dealloc, // Memory management
    .tp_methods = Fabric_methods,
};

static struct PyModuleDef pytfmbsmodule = {
    PyModuleDef_HEAD_INIT,
    "pytfmbs",
    "Python interface for the Ternary Fabric",
    -1, NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_pytfmbs(void) {
    PyObject *m;
    if (PyType_Ready(&FabricType) < 0) return NULL;
    m = PyModule_Create(&pytfmbsmodule);
    if (m == NULL) return NULL;
    Py_INCREF(&FabricType);
    if (PyModule_AddObject(m, "Fabric", (PyObject *)&FabricType) < 0) {
        Py_DECREF(&FabricType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}