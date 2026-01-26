#include <Python.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h> // Required for calloc/free
#include "tfmbs.h"

typedef struct {
    PyObject_HEAD
    void* mmio_ptr;
    int is_mock;
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
    self->is_mock = 1;
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
        self->is_mock = 1;
        if (!self->mmio_ptr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate mock MMIO buffer");
            return -1;
        }
        ((uint32_t*)self->mmio_ptr)[1] = 0x2;
        return 0;
    }
    self->mmio_ptr = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x40000000);
    self->is_mock = 0;
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
    char* filename = NULL;
    int is_file = 0;

    // Try parsing (offset, bytes)
    if (!PyArg_ParseTuple(args, "Iy*", &offset, &buffer)) {
        PyErr_Clear();
        // Try parsing (filename, offset)
        if (!PyArg_ParseTuple(args, "sI", &filename, &offset)) {
            return NULL;
        }
        is_file = 1;
        FILE *f = fopen(filename, "rb");
        if (!f) {
            PyErr_SetString(PyExc_IOError, "Could not open file");
            return NULL;
        }
        fseek(f, 0, SEEK_END);
        size_t fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        void* file_data = malloc(fsize);
        if (!file_data) {
            fclose(f);
            return PyErr_NoMemory();
        }
        if (fread(file_data, 1, fsize, f) != fsize) {
            free(file_data);
            fclose(f);
            PyErr_SetString(PyExc_IOError, "Read error");
            return NULL;
        }
        fclose(f);
        buffer.buf = file_data;
        buffer.len = fsize;
        buffer.obj = NULL;
    }

    if (!self->mmio_ptr) {
        if (!is_file) PyBuffer_Release(&buffer); else free(buffer.buf);
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    if (offset + buffer.len > 65536) {
        if (!is_file) PyBuffer_Release(&buffer); else free(buffer.buf);
        PyErr_SetString(PyExc_ValueError, "Write out of bounds");
        return NULL;
    }

    volatile uint32_t* target = (uint32_t*)((uint8_t*)self->mmio_ptr + offset);
    uint32_t* source = (uint32_t*)buffer.buf;
    size_t count = buffer.len / 4;

    for (size_t i = 0; i < count; i++) {
        target[i] = source[i];
    }

#if defined(__APPLE__) || defined(MOCK_MODE)
    printf("[TFMBS-Mock] Loaded %zu words into 0x%x %s\n", count, offset, is_file ? filename : "");
#endif

    if (!is_file) PyBuffer_Release(&buffer); else free(buffer.buf);
    Py_RETURN_NONE;
}

static PyObject* Fabric_profile(FabricObject *self, PyObject *args) {
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    volatile uint32_t* regs = (uint32_t*)self->mmio_ptr;
    PyObject* dict = PyDict_New();

    // Profiling counters start at offset 0x20 -> regs[8]
    PyDict_SetItemString(dict, "cycles", PyLong_FromUnsignedLong(regs[8]));
    PyDict_SetItemString(dict, "utilization", PyLong_FromUnsignedLong(regs[9]));

    PyObject* skips = PyList_New(15);
    for (int i = 0; i < 15; i++) {
        // Skip counters start at 0x28 -> regs[10]
        PyList_SetItem(skips, i, PyLong_FromUnsignedLong(regs[10 + i]));
    }
    PyDict_SetItemString(dict, "skips", skips);

    return dict;
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
    PyObject *tfd_dict;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &tfd_dict))
        return NULL;

    volatile uint32_t* regs = (uint32_t*)self->mmio_ptr;
    if (!regs) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    PyObject *item;

    // base_addr
    if ((item = PyDict_GetItemString(tfd_dict, "base_addr"))) {
        regs[2] = (uint32_t)PyLong_AsUnsignedLong(item);
    }

    // frame_len or depth
    if ((item = PyDict_GetItemString(tfd_dict, "depth")) ||
        (item = PyDict_GetItemString(tfd_dict, "frame_len"))) {
        regs[3] = (uint32_t)PyLong_AsLong(item);
    }

    // lane_stride or stride
    if ((item = PyDict_GetItemString(tfd_dict, "lane_stride")) ||
        (item = PyDict_GetItemString(tfd_dict, "stride"))) {
        regs[4] = (uint32_t)PyLong_AsLong(item);
    }

    // exec_hints
    if ((item = PyDict_GetItemString(tfd_dict, "exec_hints"))) {
        regs[5] = (uint32_t)PyLong_AsUnsignedLong(item);
    }

    // lane_count
    if ((item = PyDict_GetItemString(tfd_dict, "lane_count"))) {
        regs[6] = (uint32_t)PyLong_AsLong(item);
    }

    // lane_mask
    if ((item = PyDict_GetItemString(tfd_dict, "lane_mask"))) {
        regs[7] = (uint32_t)PyLong_AsLong(item);
    } else {
        regs[7] = 0x7FFF; // Default mask (15 lanes)
    }
    
#ifdef __APPLE__
    printf("[TFMBS-Mock] Frame started via TFD\n");
#endif

    regs[0] = 0x1;                 // Start signal

    if (self->is_mock) {
    // --- Mock Simulation Logic ---
    uint32_t depth = regs[3];
    uint32_t exec_hints = regs[5];
    uint32_t lane_count = regs[6];
    uint32_t lane_mask = regs[7];

    // SRAM regions at 0x1000 and 0x2000 (word-indexed)
    uint32_t* weight_sram = (uint32_t*)((uint8_t*)regs + 0x1000);
    uint32_t* input_sram = (uint32_t*)((uint8_t*)regs + 0x2000);
    uint32_t* results = (uint32_t*)((uint8_t*)regs + 0x100);
    uint32_t* cycle_count = &regs[8];
    uint32_t* utilization_count = &regs[9];
    uint32_t* skip_counts = &regs[10];

    // Reset results and counters
    for (int i=0; i<15; i++) {
        results[i] = 0;
        skip_counts[i] = 0;
    }
    *cycle_count = 0;
    *utilization_count = 0;

    for (uint32_t d = 0; d < depth; d++) {
        (*cycle_count)++;
        uint32_t w_word = weight_sram[d];
        uint32_t i_word = input_sram[d];

        int8_t w_trits[15];
        int8_t i_trits[15];

        uint32_t temp_w = w_word;
        uint32_t temp_i = i_word;
        for (int b = 0; b < 3; b++) {
            uint8_t wb = temp_w & 0xFF; temp_w >>= 8;
            uint8_t ib = temp_i & 0xFF; temp_i >>= 8;
            for (int p = 0; p < 5; p++) {
                uint8_t wt = wb % 3; wb /= 3;
                uint8_t it = ib % 3; ib /= 3;
                w_trits[b*5 + p] = (wt == 2) ? -1 : (int8_t)wt;
                i_trits[b*5 + p] = (it == 2) ? -1 : (int8_t)it;
            }
        }

        int active_this_cycle = 0;
        for (int l = 0; l < 15; l++) {
            if (l < lane_count && (lane_mask & (1 << l))) {
                active_this_cycle++;
                int8_t w = w_trits[l];
                int8_t i = i_trits[l];

                if ((exec_hints & (1 << 17)) && (w == 0 || i == 0)) {
                    skip_counts[l]++;
                } else {
                    results[l] += (w * i);
                }
            }
        }
        *utilization_count += active_this_cycle;
    }
    regs[1] |= 0x2; // Set Done bit
    }

    // In hardware, we wait for the RTL to toggle bit 1 of regs[1].
    while(!(regs[1] & 0x2));       

    Py_RETURN_NONE;
}

static PyMethodDef Fabric_methods[] = {
    {"run", (PyCFunction)Fabric_run, METH_VARARGS, "Execute a Ternary Frame Descriptor"},
    {"load", (PyCFunction)Fabric_load, METH_VARARGS, "Load binary data into Fabric SRAM (filename, offset) or (offset, bytes)"},
    {"results", (PyCFunction)Fabric_results, METH_NOARGS, "Read accumulated results from the Fabric"},
    {"profile", (PyCFunction)Fabric_profile, METH_NOARGS, "Read performance counters from the Fabric"},
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