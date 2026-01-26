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
#ifdef __APPLE__
        free(self->mmio_ptr); // Free mock memory
#else
        munmap(self->mmio_ptr, 4096); // Unmap real hardware
#endif
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// 2. Logic Implementations
static int Fabric_init(FabricObject *self, PyObject *args, PyObject *kwds) {
#ifdef __APPLE__
    // macOS Mock Mode: Allocate 4KB of RAM to act as registers
    printf("[TFMBS] Initializing macOS Mock Mode (Virtual Registers)\n");
    self->mmio_ptr = calloc(1, 4096); 
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
        PyErr_SetString(PyExc_RuntimeError, "Could not open /dev/mem. Root required.");
        return -1;
    }
    self->mmio_ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x40000000);
    close(fd);
    if (self->mmio_ptr == MAP_FAILED) {
        PyErr_SetString(PyExc_RuntimeError, "mmap failed.");
        return -1;
    }
#endif
    return 0;
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