#pragma once
#include <amdgpu.h>
#include <fcntl.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

static void* drm_alloc_vram(size_t bytes) {
    static int fd = -1;
    static amdgpu_device_handle dev;
    if (fd == -1) {
        fd = open("/dev/dri/renderD129", O_RDWR);
        if (fd < 0) fd = open("/dev/dri/renderD128", O_RDWR);
        if (fd < 0) return nullptr;
        uint32_t major, minor;
        if (amdgpu_device_initialize(fd, &major, &minor, &dev) != 0) return nullptr;
    }
    
    struct amdgpu_bo_alloc_request req = {};
    req.alloc_size = bytes;
    req.phys_alignment = 4096;
    req.preferred_heap = 0x4; // AMDGPU_GEM_DOMAIN_VRAM
    req.flags = (1 << 0); // CPU_ACCESS_REQUIRED
    
    amdgpu_bo_handle bo;
    if (amdgpu_bo_alloc(dev, &req, &bo) != 0) {
        return nullptr;
    }
    
    uint32_t dmabuf_fd;
    if (amdgpu_bo_export(bo, amdgpu_bo_handle_type_dma_buf_fd, &dmabuf_fd) != 0) return nullptr;
    
    hipExternalMemoryHandleDesc desc = {};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = dmabuf_fd;
    desc.size = bytes;
    
    hipExternalMemory_t extMem;
    if (hipImportExternalMemory(&extMem, &desc) != hipSuccess) return nullptr;
    
    hipExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = bytes;
    void *dev_ptr = nullptr;
    if (hipExternalMemoryGetMappedBuffer(&dev_ptr, extMem, &bufDesc) != hipSuccess) return nullptr;
    return dev_ptr;
}

#define HIP_MALLOC_VRAM(ptr, size) \
    do { \
        *(ptr) = (__typeof__(*(ptr)))drm_alloc_vram(size); \
        if (!*(ptr)) HIP_CHECK(hipMalloc(ptr, size)); \
    } while(0)
