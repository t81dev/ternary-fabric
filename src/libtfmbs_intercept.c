#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include "tfmbs_device.h"

static void* (*real_malloc)(size_t) = NULL;
static void (*real_free)(void*) = NULL;
static void* (*real_realloc)(void*, size_t) = NULL;
static void* (*real_memcpy)(void*, const void*, size_t) = NULL;
static void* (*real_memset)(void*, int, size_t) = NULL;
static void* (*real_mmap)(void*, size_t, int, int, int, off_t) = NULL;

static __thread int in_interposer = 0;
static int initializing = 0;
static char tmp_alloc_buf[16384];
static size_t tmp_alloc_used = 0;

#define FABRIC_THRESHOLD (1024 * 1024)

static void init_intercept() {
    if (real_malloc || initializing) return;
    initializing = 1;

    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_free = dlsym(RTLD_NEXT, "free");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_memcpy = dlsym(RTLD_NEXT, "memcpy");
    real_memset = dlsym(RTLD_NEXT, "memset");
    real_mmap = dlsym(RTLD_NEXT, "mmap");

    initializing = 0;
}

void* malloc(size_t size) {
    if (!real_malloc) {
        init_intercept();
        if (!real_malloc) {
            // dlsym is calling malloc during initialization
            if (tmp_alloc_used + size < sizeof(tmp_alloc_buf)) {
                void* ptr = tmp_alloc_buf + tmp_alloc_used;
                tmp_alloc_used += size;
                return ptr;
            }
            return NULL;
        }
    }

    if (in_interposer) return real_malloc(size);

    if (size >= FABRIC_THRESHOLD) {
        in_interposer = 1;
        void* ptr = fabric_alloc(size);
        in_interposer = 0;
        if (ptr) {
            fprintf(stderr, "[TFMBS-Intercept] Redirected malloc(%zu) -> Fabric: %p\n", size, ptr);
            return ptr;
        }
    }
    return real_malloc(size);
}

void free(void* ptr) {
    if (!ptr) return;
    if (ptr >= (void*)tmp_alloc_buf && ptr < (void*)(tmp_alloc_buf + sizeof(tmp_alloc_buf))) {
        return;
    }

    if (!real_free) init_intercept();

    if (is_fabric_ptr(ptr)) {
        fabric_free(ptr);
        return;
    }
    if (real_free) real_free(ptr);
}

void* realloc(void* ptr, size_t size) {
    if (!real_realloc) init_intercept();
    if (in_interposer || !real_realloc) return real_realloc ? real_realloc(ptr, size) : NULL;

    if (is_fabric_ptr(ptr)) {
        fprintf(stderr, "[TFMBS-Intercept] WARNING: realloc on Fabric pointer %p\n", ptr);
        void* new_ptr = malloc(size);
        return new_ptr;
    }
    return real_realloc(ptr, size);
}

void* memcpy(void* dest, const void* src, size_t n) {
    if (!real_memcpy) init_intercept();
    if (in_interposer || !real_memcpy) return real_memcpy ? real_memcpy(dest, src, n) : __builtin_memcpy(dest, src, n);

    if (is_fabric_ptr(dest)) {
        in_interposer = 1;
        fabric_memcpy_to(dest, src, n, 0);
        in_interposer = 0;
        return dest;
    }
    if (is_fabric_ptr(src)) {
        in_interposer = 1;
        fabric_memcpy_from(dest, src, n, 0);
        in_interposer = 0;
        return dest;
    }
    return real_memcpy(dest, src, n);
}

void* memset(void* s, int c, size_t n) {
    if (!real_memset) init_intercept();
    if (in_interposer || !real_memset) return real_memset ? real_memset(s, c, n) : __builtin_memset(s, c, n);

    if (is_fabric_ptr(s)) {
        in_interposer = 1;
        real_memset(s, c, n);
        in_interposer = 0;
        return s;
    }
    return real_memset(s, c, n);
}

void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (!real_mmap) init_intercept();
    if (in_interposer || !real_mmap) return real_mmap ? real_mmap(addr, length, prot, flags, fd, offset) : MAP_FAILED;

    if (length >= FABRIC_THRESHOLD && (flags & MAP_ANONYMOUS)) {
        in_interposer = 1;
        void* ptr = fabric_alloc(length);
        in_interposer = 0;
        if (ptr) {
            fprintf(stderr, "[TFMBS-Intercept] Redirected mmap(%zu) -> Fabric: %p\n", length, ptr);
            return ptr;
        }
    }
    return real_mmap(addr, length, prot, flags, fd, offset);
}
