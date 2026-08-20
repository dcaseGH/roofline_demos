#pragma once

#include <cstdlib>
#include <new>
#include <vector>

// Custom C++17 allocator for aligned vectors.
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        std::size_t remainder = bytes % Alignment;
        if (remainder != 0) {
            bytes += Alignment - remainder;
        }

        void* ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* pointer, std::size_t) noexcept {
        std::free(pointer);
    }
};

using AlignedVectorFloat = std::vector<float, AlignedAllocator<float, 32>>;
