#ifndef JETSTREAM_MEMORY_CUDA_BUFFER_HH
#define JETSTREAM_MEMORY_CUDA_BUFFER_HH

#include <memory>

#include "jetstream/memory/devices/base/buffer.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/memory/devices/vulkan/buffer.hh"
#endif

namespace Jetstream {

template<>
class TensorBuffer<Device::CUDA> {
 public:
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const bool& host_accessible = false);

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer);
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    explicit TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                          const TensorPrototypeMetadata& prototype,
                          const std::shared_ptr<TensorBuffer<Device::Metal>>& root_buffer);
#endif

    ~TensorBuffer();

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    constexpr const bool& host_accessible() const {
        return _host_accessible;
    }

    constexpr const void* data() const noexcept {
        return _buffer;
    }

    constexpr void* data() noexcept {
        return _buffer;
    }

 private:
    void* _buffer;
    bool owns_data = false;
    bool _host_accessible = false;
    Device external_memory_device = Device::None;

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    int vulkan_file_descriptor = 0;
    CUexternalMemory vulkan_external_memory = nullptr;
#endif
};

}  // namespace Jetstream

#endif
