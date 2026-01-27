#include <Python.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h> // Required for calloc/free
#include "tfmbs.h"

#define MOCK_SRAM_SIZE 1048576

typedef struct {
    PyObject_HEAD
    void* mmio_ptr;
    int is_mock;
} FabricObject;

// 1. Cleanup handler for the Python Object
static void Fabric_dealloc(FabricObject *self) {
    if (self->mmio_ptr) {
        if (self->is_mock) {
            free(self->mmio_ptr); // Free mock memory
        } else {
#if defined(__APPLE__) || defined(MOCK_MODE)
            free(self->mmio_ptr); // Should not happen if is_mock is correct, but for safety
#else
            munmap(self->mmio_ptr, 65536); // Unmap real hardware
#endif
        }
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// 2. Logic Implementations
static int Fabric_init(FabricObject *self, PyObject *args, PyObject *kwds) {
#if defined(__APPLE__) || defined(MOCK_MODE)
    // Mock Mode: Allocate 1MB of RAM to act as registers and SRAM
    printf("[TFMBS] Initializing Mock Mode (Virtual Registers + SRAM, 1MB)\n");
    self->mmio_ptr = calloc(1, MOCK_SRAM_SIZE);
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
        printf("[TFMBS] /dev/mem not accessible. Falling back to Mock Mode (1MB).\n");
        self->mmio_ptr = calloc(1, MOCK_SRAM_SIZE);
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

static PyObject* Fabric_load_stream(FabricObject *self, PyObject *args) {
    PyObject *tfd_dict;
    Py_buffer buffer;

    if (!PyArg_ParseTuple(args, "O!y*", &PyDict_Type, &tfd_dict, &buffer))
        return NULL;

    if (!self->mmio_ptr) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    if (self->is_mock) {
        printf("[TFMBS-Mock] AXI-Stream DMA Transfer Started (size=%zu)\n", buffer.len);
        volatile uint32_t* regs = (uint32_t*)self->mmio_ptr;

        // Parse TFD header and simulate hardware DMA load
        PyObject *item;
        uint32_t base_addr = 0;
        if ((item = PyDict_GetItemString(tfd_dict, "base_addr"))) base_addr = (uint32_t)PyLong_AsUnsignedLong(item);

        size_t limit = self->is_mock ? MOCK_SRAM_SIZE : 65536;
        if ((size_t)base_addr + (size_t)buffer.len > limit) {
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_ValueError, "Write out of bounds");
            return NULL;
        }

        // Mock SRAM load
        uint8_t* target = (uint8_t*)self->mmio_ptr + base_addr;
        memcpy(target, buffer.buf, buffer.len);

        // Increment burst wait cycles in mock to simulate transfer time
        regs[26] += (uint32_t)(buffer.len / 4);
    } else {
        // In real hardware, we would write to the AXI-Stream FIFO or DMA controller
    }

    PyBuffer_Release(&buffer);
    Py_RETURN_NONE;
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

    size_t limit = self->is_mock ? MOCK_SRAM_SIZE : 65536;
    if ((size_t)offset + (size_t)buffer.len > limit) {
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

    // Mock Broadcast Support (0x9000)
    if (self->is_mock && offset == 0x9000) {
        for (int t = 0; t < 4; t++) {
            volatile uint32_t* t_target = (uint32_t*)((uint8_t*)self->mmio_ptr + 0x1000 + t * 0x2000);
            memcpy((void*)t_target, buffer.buf, buffer.len);
        }
    }

#if defined(__APPLE__) || defined(MOCK_MODE)
    printf("[TFMBS-Mock] Loaded %zu words into 0x%x %s\n", count, offset, is_file ? filename : "");
#endif

    if (!is_file) PyBuffer_Release(&buffer); else free(buffer.buf);
    Py_RETURN_NONE;
}

static PyObject* Fabric_profile_tile(FabricObject *self, PyObject *args) {
    int tile_id;
    if (!PyArg_ParseTuple(args, "i", &tile_id)) return NULL;
    if (tile_id < 0 || tile_id >= 4) {
        PyErr_SetString(PyExc_ValueError, "Invalid tile_id");
        return NULL;
    }
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
    PyObject* dict = PyDict_New();

    uint32_t skip_off = (tile_id == 0) ? 0x28 : (0x200 + (tile_id-1)*0x100 + 0x28);
    uint32_t active_off = (tile_id == 0) ? 0x70 : (0x200 + (tile_id-1)*0x100 + 0x70);
    uint32_t overflow_off = (tile_id == 0) ? 0x6C : (0x200 + (tile_id-1)*0x100 + 0x6C);

    volatile uint32_t* skips_ptr = (volatile uint32_t*)((uint8_t*)regs + skip_off);
    volatile uint32_t* active_ptr = (volatile uint32_t*)((uint8_t*)regs + active_off);
    volatile uint32_t* overflow_ptr = (volatile uint32_t*)((uint8_t*)regs + overflow_off);

    PyObject* skips = PyList_New(15);
    PyObject* active = PyList_New(15);
    for (int i = 0; i < 15; i++) {
        PyList_SetItem(skips, i, PyLong_FromUnsignedLong(skips_ptr[i]));
        PyList_SetItem(active, i, PyLong_FromUnsignedLong(active_ptr[i]));
    }
    PyDict_SetItemString(dict, "skips", skips);
    PyDict_SetItemString(dict, "active_cycles", active);
    PyDict_SetItemString(dict, "overflow_flags", PyLong_FromUnsignedLong(*overflow_ptr & 0x7FFF));

    return dict;
}

static PyObject* Fabric_profile(FabricObject *self, PyObject *args) {
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
    PyObject* dict = PyDict_New();

    PyDict_SetItemString(dict, "cycles", PyLong_FromUnsignedLong(regs[8]));
    PyDict_SetItemString(dict, "utilization", PyLong_FromUnsignedLong(regs[9]));

    PyObject* skips = PyList_New(15);
    for (int i = 0; i < 15; i++) {
        uint32_t total_skips = 0;
        for (int t = 0; t < 4; t++) {
            uint32_t skip_off = (t == 0) ? 0x28 : (0x200 + (t-1)*0x100 + 0x28);
            volatile uint32_t* skips_ptr = (volatile uint32_t*)((uint8_t*)regs + skip_off);
            total_skips += skips_ptr[i];
        }
        PyList_SetItem(skips, i, PyLong_FromUnsignedLong(total_skips));
    }
    PyDict_SetItemString(dict, "skips", skips);

    return dict;
}

static PyObject* Fabric_profile_detailed(FabricObject *self, PyObject *args) {
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
    PyObject* dict = Fabric_profile(self, NULL);

    PyDict_SetItemString(dict, "burst_wait_cycles", PyLong_FromUnsignedLong(regs[26]));

    uint32_t total_overflow = 0;
    PyObject* active = PyList_New(15);
    for (int i = 0; i < 15; i++) {
        uint32_t total_active = 0;
        for (int t = 0; t < 4; t++) {
            uint32_t active_off = (t == 0) ? 0x70 : (0x200 + (t-1)*0x100 + 0x70);
            volatile uint32_t* active_ptr = (volatile uint32_t*)((uint8_t*)regs + active_off);
            total_active += active_ptr[i];

            if (i == 0) {
                uint32_t overflow_off = (t == 0) ? 0x6C : (0x200 + (t-1)*0x100 + 0x6C);
                volatile uint32_t* overflow_ptr = (volatile uint32_t*)((uint8_t*)regs + overflow_off);
                total_overflow |= (*overflow_ptr & 0x7FFF);
            }
        }
        PyList_SetItem(active, i, PyLong_FromUnsignedLong(total_active));
    }
    PyDict_SetItemString(dict, "overflow_flags", PyLong_FromUnsignedLong(total_overflow));
    PyDict_SetItemString(dict, "active_cycles", active);

    return dict;
}

static PyObject* Fabric_results(FabricObject *self, PyObject *args) {
    int tile_id = 0;
    if (!PyArg_ParseTuple(args, "|i", &tile_id)) return NULL;

    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    if (tile_id < -1 || tile_id >= 4) {
        PyErr_SetString(PyExc_ValueError, "Invalid tile_id");
        return NULL;
    }

    if (tile_id == -1) {
        PyObject* list = PyList_New(60);
        volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
        for (int t = 0; t < 4; t++) {
            volatile uint32_t* t_results = (volatile uint32_t*)((uint8_t*)regs + 0x100 + t * 0x40);
            for (int i = 0; i < 15; i++) {
                PyList_SetItem(list, t * 15 + i, PyLong_FromLong((int32_t)t_results[i]));
            }
        }
        return list;
    } else {
        PyObject* list = PyList_New(15);
        volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
        volatile uint32_t* t_results = (volatile uint32_t*)((uint8_t*)regs + 0x100 + tile_id * 0x40);
        for (int i = 0; i < 15; i++) {
            PyList_SetItem(list, i, PyLong_FromLong((int32_t)t_results[i]));
        }
        return list;
    }
}

static int internal_Fabric_submit(FabricObject *self, PyObject *tfd_dict) {
    volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
    PyObject *item;
    if ((item = PyDict_GetItemString(tfd_dict, "base_addr"))) regs[2] = (uint32_t)PyLong_AsUnsignedLong(item); else regs[2] = 0;
    if ((item = PyDict_GetItemString(tfd_dict, "depth")) || (item = PyDict_GetItemString(tfd_dict, "frame_len"))) regs[3] = (uint32_t)PyLong_AsLong(item); else regs[3] = 0;
    if ((item = PyDict_GetItemString(tfd_dict, "lane_stride")) || (item = PyDict_GetItemString(tfd_dict, "stride"))) regs[4] = (uint32_t)PyLong_AsLong(item); else regs[4] = 1;
    if ((item = PyDict_GetItemString(tfd_dict, "exec_hints"))) regs[5] = (uint32_t)PyLong_AsUnsignedLong(item); else regs[5] = 0;
    if ((item = PyDict_GetItemString(tfd_dict, "lane_count"))) regs[6] = (uint32_t)PyLong_AsLong(item); else regs[6] = 15;
    if ((item = PyDict_GetItemString(tfd_dict, "lane_mask"))) regs[7] = (uint32_t)PyLong_AsLong(item); else regs[7] = 0x7FFF;

    uint32_t tile_mask = 0x1;
    if ((item = PyDict_GetItemString(tfd_dict, "tile_mask"))) tile_mask = (uint32_t)PyLong_AsLong(item);
    
    regs[0] = 0x1 | (tile_mask << 8);
    regs[1] &= ~0x2;

    if (self->is_mock) {
        uint32_t depth = regs[3];
        uint32_t exec_hints = regs[5];
        uint32_t lane_count = regs[6];
        uint32_t lane_mask = regs[7];
        uint8_t  op_mode = exec_hints & 0xFF;
        uint32_t stride = regs[4];
        if (stride == 0) stride = 1;

        regs[8] = 0; // cycles
        regs[9] = 0; // utilization
        regs[26] = depth / 10; // burst_wait

        for (int t = 0; t < 4; t++) {
            uint32_t skip_off = (t == 0) ? 0x28 : (0x200 + (t-1)*0x100 + 0x28);
            uint32_t active_off = (t == 0) ? 0x70 : (0x200 + (t-1)*0x100 + 0x70);
            uint32_t overflow_off = (t == 0) ? 0x6C : (0x200 + (t-1)*0x100 + 0x6C);
            volatile uint32_t* t_results = (volatile uint32_t*)((uint8_t*)regs + 0x100 + t * 0x40);
            volatile uint32_t* t_skips = (volatile uint32_t*)((uint8_t*)regs + skip_off);
            volatile uint32_t* t_active = (volatile uint32_t*)((uint8_t*)regs + active_off);
            volatile uint32_t* t_overflow = (volatile uint32_t*)((uint8_t*)regs + overflow_off);

            for (int i=0; i<15; i++) {
                // State Management: Only clear accumulator if BIAS_EN hint is NOT set
                if (!(exec_hints & TFMBS_HINT_BIAS_EN)) {
                    t_results[i] = (op_mode == TFMBS_KERNEL_MAXPOOL && (exec_hints >> 29) == 0x1) ? 0x7FFFFFFF :
                                 (op_mode == TFMBS_KERNEL_MAXPOOL && (exec_hints >> 29) == 0x0) ? (uint32_t)0x80000000 : 0;
                }
                t_skips[i] = 0;
                t_active[i] = 0;
            }
            *t_overflow = 0;
        }

        for (uint32_t d = 0; d < depth; d++) {
            regs[8]++;
            int total_active_this_cycle = 0;

            for (int t = 0; t < 4; t++) {
                if (!(tile_mask & (1 << t))) continue;

                volatile uint32_t* t_weight_sram = (volatile uint32_t*)((uint8_t*)regs + 0x1000 + t * 0x2000);
                volatile uint32_t* t_input_sram = (volatile uint32_t*)((uint8_t*)regs + 0x2000 + t * 0x2000);
                volatile uint32_t* t_results = (volatile uint32_t*)((uint8_t*)regs + 0x100 + t * 0x40);
                uint32_t skip_off = (t == 0) ? 0x28 : (0x200 + (t-1)*0x100 + 0x28);
                uint32_t active_off = (t == 0) ? 0x70 : (0x200 + (t-1)*0x100 + 0x70);
                uint32_t overflow_off = (t == 0) ? 0x6C : (0x200 + (t-1)*0x100 + 0x6C);
                volatile uint32_t* t_skips = (volatile uint32_t*)((uint8_t*)regs + skip_off);
                volatile uint32_t* t_active = (volatile uint32_t*)((uint8_t*)regs + active_off);
                volatile uint32_t* t_overflow = (volatile uint32_t*)((uint8_t*)regs + overflow_off);

                uint32_t idx = d * stride;
                if (op_mode == TFMBS_KERNEL_CONV2D) {
                    uint32_t conv_stride = ((exec_hints >> 20) & 0x3) + 1;
                    idx = d * stride * conv_stride;
                } else if (op_mode == TFMBS_KERNEL_CONV3D) {
                    uint32_t conv_stride = ((exec_hints >> 20) & 0x3) + 1;
                    idx = d * stride * conv_stride * conv_stride;
                }
                if (idx * 4 + (uint8_t*)t_input_sram - (uint8_t*)regs >= MOCK_SRAM_SIZE) continue;

                uint32_t w_word = t_weight_sram[d];
                uint32_t i_word = t_input_sram[idx];

                int8_t w_trits[15], i_trits[15];
                uint32_t temp_w = w_word, temp_i = i_word;
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

                for (uint32_t l = 0; l < 15; l++) {
                    if (l < lane_count && (lane_mask & (1 << l))) {
                        total_active_this_cycle++;
                        t_active[l]++;
                        int8_t w = w_trits[l], i = i_trits[l];

                        if ((exec_hints & (1 << 17)) && (w == 0 || i == 0)) {
                            t_skips[l]++;
                        } else {
                            int32_t prod = (int32_t)w * (int32_t)i;
                            if (op_mode == TFMBS_KERNEL_DOT || op_mode == TFMBS_KERNEL_TGEMM ||
                                op_mode == TFMBS_KERNEL_CONV2D || op_mode == TFMBS_KERNEL_CONV3D ||
                                op_mode == TFMBS_KERNEL_LSTM || op_mode == TFMBS_KERNEL_ATTN) {
                                int32_t old_res = (int32_t)t_results[l];
                                t_results[l] += prod;
                                if (prod > 0 && old_res > 0 && (int32_t)t_results[l] < 0) *t_overflow |= (1 << l);
                                if (prod < 0 && old_res < 0 && (int32_t)t_results[l] > 0) *t_overflow |= (1 << l);
                            } else if (op_mode == TFMBS_KERNEL_MUL) {
                                t_results[l] = prod;
                            } else if (op_mode == TFMBS_KERNEL_MAXPOOL) {
                                uint32_t pool_op = (exec_hints >> 29) & 0x3;
                                if (pool_op == 0x0 && prod > (int32_t)t_results[l]) t_results[l] = prod;
                                else if (pool_op == 0x1 && prod < (int32_t)t_results[l]) t_results[l] = prod;
                                else if (pool_op == 0x2) t_results[l] += prod;
                            }
                        }
                    }
                }
            }
            regs[9] += total_active_this_cycle;
        }
        regs[1] |= 0x2;
    }
    return 0;
}

static PyObject* Fabric_submit(FabricObject *self, PyObject *args) {
    PyObject *tfd_dict;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &tfd_dict))
        return NULL;
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }
    internal_Fabric_submit(self, tfd_dict);
    Py_RETURN_NONE;
}

static PyObject* Fabric_is_done(FabricObject *self, PyObject *args) {
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }
    volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
    if (regs[1] & 0x2) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject* Fabric_wait(FabricObject *self, PyObject *args) {
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }
    volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
    while(!(regs[1] & 0x2));
    return Fabric_profile_detailed(self, NULL);
}

static PyObject* Fabric_run(FabricObject *self, PyObject *args) {
    PyObject *tfd_dict;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &tfd_dict))
        return NULL;
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }
    internal_Fabric_submit(self, tfd_dict);
    return Fabric_wait(self, NULL);
}

static PyObject* Fabric_run_batch(FabricObject *self, PyObject *args) {
    PyObject *tfd_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &tfd_list))
        return NULL;
    if (!self->mmio_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "MMIO not initialized");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(tfd_list);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *tfd_dict = PyList_GetItem(tfd_list, i);
        if (!PyDict_Check(tfd_dict)) continue;
        internal_Fabric_submit(self, tfd_dict);
        // Sequential wait for simplicity in mock
        volatile uint32_t* regs = (volatile uint32_t*)self->mmio_ptr;
        while(!(regs[1] & 0x2));
    }

    return Fabric_profile_detailed(self, NULL);
}

static PyMethodDef Fabric_methods[] = {
    {"run", (PyCFunction)Fabric_run, METH_VARARGS, "Execute a Ternary Frame Descriptor"},
    {"run_batch", (PyCFunction)Fabric_run_batch, METH_VARARGS, "Execute a list of Ternary Frame Descriptors"},
    {"submit", (PyCFunction)Fabric_submit, METH_VARARGS, "Submit a TFD for asynchronous execution"},
    {"wait", (PyCFunction)Fabric_wait, METH_NOARGS, "Wait for the current operation to complete"},
    {"is_done", (PyCFunction)Fabric_is_done, METH_NOARGS, "Check if the current operation is complete"},
    {"load", (PyCFunction)Fabric_load, METH_VARARGS, "Load binary data into Fabric SRAM (filename, offset) or (offset, bytes)"},
    {"load_stream", (PyCFunction)Fabric_load_stream, METH_VARARGS, "Load data via AXI-Stream DMA (tfd_dict, bytes)"},
    {"results", (PyCFunction)Fabric_results, METH_VARARGS, "Read accumulated results from the Fabric"},
    {"profile", (PyCFunction)Fabric_profile, METH_NOARGS, "Read performance counters from the Fabric"},
    {"profile_detailed", (PyCFunction)Fabric_profile_detailed, METH_NOARGS, "Read detailed performance counters from the Fabric"},
    {"profile_tile", (PyCFunction)Fabric_profile_tile, METH_VARARGS, "Read performance counters for a specific tile"},
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
    .tp_dealloc = (destructor)Fabric_dealloc,
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
