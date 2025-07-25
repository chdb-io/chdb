#pragma once

#include "config.h"

#include <base/getPageSize.h>
#include <boost/noncopyable.hpp>
#include <Common/Allocator.h>
#include <Common/BitHelpers.h>
#include <Common/GWPAsan.h>
#include <Common/PODArray_fwd.h>
#include <Common/memcpySmall.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <numeric>

#ifndef NDEBUG
#include <sys/mman.h>
#endif

/** Whether we can use memcpy instead of a loop with assignment to T from U.
  * It is Ok if types are the same. And if types are integral and of the same size,
  *  example: char, signed char, unsigned char.
  * It's not Ok for int and float.
  * Don't forget to apply std::decay when using this constexpr.
  */
template <typename T, typename U>
constexpr bool memcpy_can_be_used_for_assignment = std::is_same_v<T, U>
    || (std::is_integral_v<T> && std::is_integral_v<U> && sizeof(T) == sizeof(U)); /// NOLINT(misc-redundant-expression)

namespace DB
{

/** A dynamic array for POD types.
  * Designed for a small number of large arrays (rather than a lot of small ones).
  * To be more precise - for use in ColumnVector.
  * It differs from std::vector in that it does not initialize the elements.
  *
  * Made noncopyable so that there are no accidental copies. You can copy the data using `assign` method.
  *
  * Only part of the std::vector interface is supported.
  *
  * The default constructor creates an empty object that does not allocate memory.
  * Then the memory is allocated at least initial_bytes bytes.
  *
  * If you insert elements with push_back, without making a `reserve`, then PODArray is about 2.5 times faster than std::vector.
  *
  * The template parameter `pad_right` - always allocate at the end of the array as many unused bytes.
  * Can be used to make optimistic reading, writing, copying with unaligned SIMD instructions.
  *
  * The template parameter `pad_left` - always allocate memory before 0th element of the array (rounded up to the whole number of elements)
  *  and zero initialize -1th element. It allows to use -1th element that will have value 0.
  * This gives performance benefits when converting an array of offsets to array of sizes.
  *
  * Some methods using allocator have TAllocatorParams variadic arguments.
  * These arguments will be passed to corresponding methods of TAllocator.
  * Example: pointer to Arena, that is used for allocations.
  *
  * Why Allocator is not passed through constructor, as it is done in C++ standard library?
  * Because sometimes we have many small objects, that share same allocator with same parameters,
  *  and we must avoid larger object size due to storing the same parameters in each object.
  * This is required for states of aggregate functions.
  *
  * TODO Pass alignment to Allocator.
  * TODO Allow greater alignment than alignof(T). Example: array of char aligned to page size.
  */
static constexpr size_t empty_pod_array_size = 1024;
alignas(std::max_align_t) extern const char empty_pod_array[empty_pod_array_size];

namespace PODArrayDetails
{

void protectMemoryRegion(void * addr, size_t len, int prot);

/// The amount of memory occupied by the num_elements of the elements.
size_t byte_size(size_t num_elements, size_t element_size); /// NOLINT

/// Minimum amount of memory to allocate for num_elements, including padding.
size_t minimum_memory_for_elements(size_t num_elements, size_t element_size, size_t pad_left, size_t pad_right); /// NOLINT

};

/** Base class that depend only on size of element, not on element itself.
  * You can static_cast to this class if you want to insert some data regardless to the actual type T.
  */
template <size_t ELEMENT_SIZE, size_t initial_bytes, typename TAllocator, size_t pad_right_, size_t pad_left_>
class PODArrayBase : private boost::noncopyable, private TAllocator    /// empty base optimization
{
protected:
    /// Round padding up to an whole number of elements to simplify arithmetic.
    static constexpr size_t pad_right = integerRoundUp(pad_right_, ELEMENT_SIZE);
    /// pad_left is also rounded up to 16 bytes to maintain alignment of allocated memory.
    static constexpr size_t pad_left = integerRoundUp(pad_left_, std::lcm(ELEMENT_SIZE, 16));
    /// Empty array will point to this static memory as padding and begin/end.
    static constexpr char * null = const_cast<char *>(empty_pod_array) + pad_left;

    static_assert(pad_left <= empty_pod_array_size && "Left Padding exceeds empty_pod_array_size. Is the element size too large?");
    static_assert(pad_left % ELEMENT_SIZE == 0, "pad_left must be multiple of element alignment");

    // If we are using allocator with inline memory, the minimal size of
    // array must be in sync with the size of this memory.
    static_assert(allocatorInitialBytes<TAllocator> == 0
                  || allocatorInitialBytes<TAllocator> == initial_bytes);

    char * c_start          = null;    /// Does not include pad_left.
    char * c_end            = null;
    char * c_end_of_storage = null;    /// Does not include pad_right.

    void alloc_for_num_elements(size_t num_elements) /// NOLINT
    {
        alloc(PODArrayDetails::minimum_memory_for_elements(num_elements, ELEMENT_SIZE, pad_left, pad_right));
    }

    template <typename ... TAllocatorParams>
    void alloc(size_t bytes, TAllocatorParams &&... allocator_params)
    {
        char * allocated = reinterpret_cast<char *>(TAllocator::alloc(bytes, std::forward<TAllocatorParams>(allocator_params)...));

        c_start = allocated + pad_left;
        c_end = c_start;
        c_end_of_storage = allocated + bytes - pad_right;

        if (pad_left)
            memset(c_start - ELEMENT_SIZE, 0, ELEMENT_SIZE);
    }

    void dealloc()
    {
        if (c_start == null)
            return;

        unprotect();

        TAllocator::free(c_start - pad_left, allocated_bytes());
    }

    template <typename ... TAllocatorParams>
    void realloc(size_t bytes, TAllocatorParams &&... allocator_params)
    {
        if (c_start == null)
        {
            alloc(bytes, std::forward<TAllocatorParams>(allocator_params)...);
            return;
        }

        unprotect();

        ptrdiff_t end_diff = c_end - c_start;

        char * allocated = reinterpret_cast<char *>(
            TAllocator::realloc(c_start - pad_left, allocated_bytes(), bytes, std::forward<TAllocatorParams>(allocator_params)...));

        c_start = allocated + pad_left;
        c_end = c_start + end_diff;
        c_end_of_storage = allocated + bytes - pad_right;
    }

    bool isInitialized() const
    {
        return (c_start != null) && (c_end != null) && (c_end_of_storage != null);
    }

    bool isAllocatedFromStack() const
    {
        static constexpr size_t stack_threshold = TAllocator::getStackThreshold();
        return (stack_threshold > 0) && (allocated_bytes() <= stack_threshold);
    }

    template <typename ... TAllocatorParams>
    void reserveForNextSize(TAllocatorParams &&... allocator_params)
    {
        if (empty())
        {
            // The allocated memory should be multiplication of ELEMENT_SIZE to hold the element, otherwise,
            // memory issue such as corruption could appear in edge case.
            realloc(std::max(integerRoundUp(initial_bytes, ELEMENT_SIZE),
                             PODArrayDetails::minimum_memory_for_elements(1, ELEMENT_SIZE, pad_left, pad_right)),
                    std::forward<TAllocatorParams>(allocator_params)...);
        }
        else
            realloc(allocated_bytes() * 2, std::forward<TAllocatorParams>(allocator_params)...);
    }

#ifndef NDEBUG
    /// Make memory region readonly with mprotect if it is large enough.
    /// The operation is slow and performed only for debug builds.
    void protectImpl(int prot)
    {
        static size_t PROTECT_PAGE_SIZE = ::getPageSize();

        char * left_rounded_up = reinterpret_cast<char *>((reinterpret_cast<intptr_t>(c_start) - pad_left + PROTECT_PAGE_SIZE - 1) / PROTECT_PAGE_SIZE * PROTECT_PAGE_SIZE);
        char * right_rounded_down = reinterpret_cast<char *>((reinterpret_cast<intptr_t>(c_end_of_storage) + pad_right) / PROTECT_PAGE_SIZE * PROTECT_PAGE_SIZE);

        if (right_rounded_down > left_rounded_up)
        {
            size_t length = right_rounded_down - left_rounded_up;
            PODArrayDetails::protectMemoryRegion(left_rounded_up, length, prot);
        }
    }

    /// Restore memory protection in destructor or realloc for further reuse by allocator.
    bool mprotected = false;
#endif

public:
    bool empty() const { return c_end == c_start; }
    size_t size() const { return (c_end - c_start) / ELEMENT_SIZE; }
    size_t capacity() const { return (c_end_of_storage - c_start) / ELEMENT_SIZE; }

    /// This method is safe to use only for information about memory usage.
    size_t allocated_bytes() const { return c_end_of_storage - c_start + pad_right + pad_left; } /// NOLINT

    void clear() { c_end = c_start; }

    template <typename ... TAllocatorParams>
    ALWAYS_INLINE /// Better performance in clang build, worse performance in gcc build.
    void reserve(size_t n, TAllocatorParams &&... allocator_params)
    {
        if (n > capacity())
            realloc(roundUpToPowerOfTwoOrZero(PODArrayDetails::minimum_memory_for_elements(n, ELEMENT_SIZE, pad_left, pad_right)), std::forward<TAllocatorParams>(allocator_params)...);
    }

    template <typename ... TAllocatorParams>
    void reserve_exact(size_t n, TAllocatorParams &&... allocator_params) /// NOLINT
    {
        if (n > capacity())
            realloc(PODArrayDetails::minimum_memory_for_elements(n, ELEMENT_SIZE, pad_left, pad_right), std::forward<TAllocatorParams>(allocator_params)...);
    }

    template <typename ... TAllocatorParams>
    void resize(size_t n, TAllocatorParams &&... allocator_params)
    {
        reserve(n, std::forward<TAllocatorParams>(allocator_params)...);
        resize_assume_reserved(n);
    }

    template <typename ... TAllocatorParams>
    void resize_exact(size_t n, TAllocatorParams &&... allocator_params) /// NOLINT
    {
        reserve_exact(n, std::forward<TAllocatorParams>(allocator_params)...);
        resize_assume_reserved(n);
    }

    template <typename ... TAllocatorParams>
    void shrink_to_fit(TAllocatorParams &&... allocator_params)
    {
        realloc(PODArrayDetails::minimum_memory_for_elements(size(), ELEMENT_SIZE, pad_left, pad_right), std::forward<TAllocatorParams>(allocator_params)...);
    }

    void resize_assume_reserved(const size_t n) /// NOLINT
    {
        c_end = c_start + PODArrayDetails::byte_size(n, ELEMENT_SIZE);
    }

    const char * raw_data() const /// NOLINT
    {
        return c_start;
    }

    template <typename ... TAllocatorParams>
    void push_back_raw(const void * ptr, TAllocatorParams &&... allocator_params) /// NOLINT
    {
        size_t required_capacity = size() + ELEMENT_SIZE;
        if (unlikely(required_capacity > capacity()))
            reserve(required_capacity, std::forward<TAllocatorParams>(allocator_params)...);

        memcpy(c_end, ptr, ELEMENT_SIZE);
        c_end += ELEMENT_SIZE;
    }

    template <typename... TAllocatorParams>
    void append_raw(const void * ptr, size_t count, TAllocatorParams &&... allocator_params) /// NOLINT
    {
        size_t bytes_to_copy = PODArrayDetails::byte_size(count, ELEMENT_SIZE);
        size_t required_capacity = size() + bytes_to_copy;
        if (unlikely(required_capacity > capacity()))
            reserve(required_capacity, std::forward<TAllocatorParams>(allocator_params)...);

        memcpy(c_end, ptr, bytes_to_copy);
        c_end += bytes_to_copy;
    }

    void protect()
    {
#ifndef NDEBUG
        protectImpl(PROT_READ);
        mprotected = true;
#endif
    }

    void unprotect()
    {
#ifndef NDEBUG
        if (mprotected)
            protectImpl(PROT_WRITE);
        mprotected = false;
#endif
    }

    template <typename It1, typename It2>
    void assertNotIntersects(It1 from_begin [[maybe_unused]], It2 from_end [[maybe_unused]])
    {
#if !defined(NDEBUG)
        const char * ptr_begin = reinterpret_cast<const char *>(&*from_begin);
        const char * ptr_end = reinterpret_cast<const char *>(&*from_end);

        /// Also it's safe if the range is empty.
        assert(!((ptr_begin >= c_start && ptr_begin < c_end) || (ptr_end > c_start && ptr_end <= c_end)) || (ptr_begin == ptr_end));
#endif
    }

    ~PODArrayBase()
    {
        dealloc();
    }
};

/// NOLINTBEGIN(bugprone-sizeof-expression)

template <typename T, size_t initial_bytes, typename TAllocator, size_t pad_right_, size_t pad_left_>
class PODArray : public PODArrayBase<sizeof(T), initial_bytes, TAllocator, pad_right_, pad_left_>
{
protected:
    using Base = PODArrayBase<sizeof(T), initial_bytes, TAllocator, pad_right_, pad_left_>;

    T * t_start()                      { return reinterpret_cast<T *>(this->c_start); } /// NOLINT
    T * t_end()                        { return reinterpret_cast<T *>(this->c_end); } /// NOLINT

    const T * t_start() const          { return reinterpret_cast<const T *>(this->c_start); } /// NOLINT
    const T * t_end() const            { return reinterpret_cast<const T *>(this->c_end); } /// NOLINT

public:
    using value_type = T;

    /// We cannot use boost::iterator_adaptor, because it defeats loop vectorization,
    ///  see https://github.com/ClickHouse/ClickHouse/pull/9442

    using iterator = T *;
    using const_iterator = const T *;


    PODArray() = default;

    explicit PODArray(size_t n)
    {
        this->alloc_for_num_elements(n);
        this->c_end += PODArrayDetails::byte_size(n, sizeof(T));
    }

    PODArray(size_t n, const T & x)
    {
        this->alloc_for_num_elements(n);
        assign(n, x);
    }

    PODArray(const_iterator from_begin, const_iterator from_end)
    {
        this->alloc_for_num_elements(from_end - from_begin);
        insert(from_begin, from_end);
    }

    PODArray(std::initializer_list<T> il)
    {
        this->reserve(std::size(il));

        for (const auto & x : il)
        {
            this->push_back(x);
        }
    }

    PODArray(PODArray && other) noexcept
    {
        this->swap(other);
    }

    PODArray & operator=(PODArray && other) noexcept
    {
        this->swap(other);
        return *this;
    }

    T * data() { return t_start(); }
    const T * data() const { return t_start(); }

    /// The index is signed to access -1th element without pointer overflow.
    T & operator[] (ssize_t n)
    {
        /// <= size, because taking address of one element past memory range is Ok in C++ (expression like &arr[arr.size()] is perfectly valid).
        assert((n >= (static_cast<ssize_t>(pad_left_) ? -1 : 0)) && (n <= static_cast<ssize_t>(this->size())));
        return t_start()[n];
    }

    const T & operator[] (ssize_t n) const
    {
        assert((n >= (static_cast<ssize_t>(pad_left_) ? -1 : 0)) && (n <= static_cast<ssize_t>(this->size())));
        return t_start()[n];
    }

    T & front()             { return t_start()[0]; }
    T & back()              { return t_end()[-1]; }
    const T & front() const { return t_start()[0]; }
    const T & back() const  { return t_end()[-1]; }

    iterator begin()              { return t_start(); }
    iterator end()                { return t_end(); }
    const_iterator begin() const  { return t_start(); }
    const_iterator end() const    { return t_end(); }
    const_iterator cbegin() const { return t_start(); }
    const_iterator cend() const   { return t_end(); }

    /// Same as resize, but zeroes new elements.
    void resize_fill(size_t n) /// NOLINT
    {
        size_t old_size = this->size();
        if (n > old_size)
        {
            this->reserve(n);
            memset(this->c_end, 0, PODArrayDetails::byte_size(n - old_size, sizeof(T)));
        }
        this->c_end = this->c_start + PODArrayDetails::byte_size(n, sizeof(T));
    }

    void resize_fill(size_t n, const T & value) /// NOLINT
    {
        size_t old_size = this->size();
        if (n > old_size)
        {
            this->reserve(n);
            std::fill(t_end(), t_end() + n - old_size, value);
        }
        this->c_end = this->c_start + PODArrayDetails::byte_size(n, sizeof(T));
    }

    template <typename U, typename ... TAllocatorParams>
    void push_back(U && x, TAllocatorParams &&... allocator_params) /// NOLINT
    {
        if (unlikely(this->c_end + sizeof(T) > this->c_end_of_storage))
            this->reserveForNextSize(std::forward<TAllocatorParams>(allocator_params)...);

        new (reinterpret_cast<void*>(t_end())) T(std::forward<U>(x));
        this->c_end += sizeof(T);
    }

    /** This method doesn't allow to pass parameters for Allocator,
      *  and it couldn't be used if Allocator requires custom parameters.
      */
    template <typename... Args>
    void emplace_back(Args &&... args) /// NOLINT
    {
        if (unlikely(this->c_end + sizeof(T) > this->c_end_of_storage))
            this->reserveForNextSize();

        new (t_end()) T(std::forward<Args>(args)...);
        this->c_end += sizeof(T);
    }

    void pop_back() /// NOLINT
    {
        this->c_end -= sizeof(T);
    }

    /// Do not insert into the array a piece of itself. Because with the resize, the iterators on themselves can be invalidated.
    template <typename It1, typename It2, typename ... TAllocatorParams>
    void insertPrepare(It1 from_begin, It2 from_end, TAllocatorParams &&... allocator_params)
    {
        this->assertNotIntersects(from_begin, from_end);
        size_t required_capacity = this->size() + (from_end - from_begin);
        if (required_capacity > this->capacity())
            this->reserve(roundUpToPowerOfTwoOrZero(required_capacity), std::forward<TAllocatorParams>(allocator_params)...);
    }

    /// Do not insert into the array a piece of itself. Because with the resize, the iterators on themselves can be invalidated.
    template <typename It1, typename It2, typename ... TAllocatorParams>
    void insert(It1 from_begin, It2 from_end, TAllocatorParams &&... allocator_params)
    {
        insertPrepare(from_begin, from_end, std::forward<TAllocatorParams>(allocator_params)...);
        insert_assume_reserved(from_begin, from_end);
    }

    /// In contrast to 'insert' this method is Ok even for inserting from itself.
    /// Because we obtain iterators after reserving memory.
    template <typename Container, typename ... TAllocatorParams>
    void insertByOffsets(Container && rhs, size_t from_begin, size_t from_end, TAllocatorParams &&... allocator_params)
    {
        static_assert(memcpy_can_be_used_for_assignment<std::decay_t<T>, std::decay_t<decltype(rhs.front())>>);

        assert(from_end >= from_begin);
        assert(from_end <= rhs.size());

        size_t required_capacity = this->size() + (from_end - from_begin);
        if (required_capacity > this->capacity())
            this->reserve(roundUpToPowerOfTwoOrZero(required_capacity), std::forward<TAllocatorParams>(allocator_params)...);

        size_t bytes_to_copy = PODArrayDetails::byte_size(from_end - from_begin, sizeof(T));
        if (bytes_to_copy)
        {
            memcpy(this->c_end, reinterpret_cast<const void *>(rhs.begin() + from_begin), bytes_to_copy);
            this->c_end += bytes_to_copy;
        }
    }

    /// Works under assumption, that it's possible to read up to 15 excessive bytes after `from_end` and this PODArray is padded.
    template <typename It1, typename It2, typename ... TAllocatorParams>
    void insertSmallAllowReadWriteOverflow15(It1 from_begin, It2 from_end, TAllocatorParams &&... allocator_params)
    {
        static_assert(pad_right_ >= PADDING_FOR_SIMD - 1);
        static_assert(sizeof(T) == sizeof(*from_begin));
        insertPrepare(from_begin, from_end, std::forward<TAllocatorParams>(allocator_params)...);
        size_t bytes_to_copy = PODArrayDetails::byte_size(from_end - from_begin, sizeof(T));
        memcpySmallAllowReadWriteOverflow15(this->c_end, reinterpret_cast<const void *>(&*from_begin), bytes_to_copy);
        this->c_end += bytes_to_copy;
    }

    /// Do not insert into the array a piece of itself. Because with the resize, the iterators on themselves can be invalidated.
    template <typename It1, typename It2>
    void insert(iterator it, It1 from_begin, It2 from_end)
    {
        static_assert(memcpy_can_be_used_for_assignment<std::decay_t<T>, std::decay_t<decltype(*from_begin)>>);

        size_t bytes_to_copy = PODArrayDetails::byte_size(from_end - from_begin, sizeof(T));
        if (!bytes_to_copy)
            return;

        size_t bytes_to_move = PODArrayDetails::byte_size(end() - it, sizeof(T));

        insertPrepare(from_begin, from_end);

        if (unlikely(bytes_to_move))
            memmove(this->c_end + bytes_to_copy - bytes_to_move, this->c_end - bytes_to_move, bytes_to_move);

        memcpy(this->c_end - bytes_to_move, reinterpret_cast<const void *>(&*from_begin), bytes_to_copy);

        this->c_end += bytes_to_copy;
    }

    template <typename ... TAllocatorParams>
    void insertFromItself(iterator from_begin, iterator from_end, TAllocatorParams && ... allocator_params)
    {
        static_assert(memcpy_can_be_used_for_assignment<std::decay_t<T>, std::decay_t<decltype(*from_begin)>>);

        /// Convert iterators to indexes because reserve can invalidate iterators
        size_t start_index = from_begin - begin();
        size_t end_index = from_end - begin();
        size_t copy_size = end_index - start_index;

        assert(start_index <= end_index);

        size_t required_capacity = this->size() + copy_size;
        if (required_capacity > this->capacity())
            this->reserve(roundUpToPowerOfTwoOrZero(required_capacity), std::forward<TAllocatorParams>(allocator_params)...);

        size_t bytes_to_copy = PODArrayDetails::byte_size(copy_size, sizeof(T));
        if (bytes_to_copy)
        {
            auto begin = this->c_start + PODArrayDetails::byte_size(start_index, sizeof(T));
            memcpy(this->c_end, reinterpret_cast<const void *>(&*begin), bytes_to_copy);
            this->c_end += bytes_to_copy;
        }
    }

    template <typename It1, typename It2>
    void insert_assume_reserved(It1 from_begin, It2 from_end) /// NOLINT
    {
        static_assert(memcpy_can_be_used_for_assignment<std::decay_t<T>, std::decay_t<decltype(*from_begin)>>);
        this->assertNotIntersects(from_begin, from_end);

        size_t bytes_to_copy = PODArrayDetails::byte_size(from_end - from_begin, sizeof(T));
        if (bytes_to_copy)
        {
            memcpy(this->c_end, reinterpret_cast<const void *>(&*from_begin), bytes_to_copy);
            this->c_end += bytes_to_copy;
        }
    }

    template <typename... TAllocatorParams>
    void swap(PODArray & rhs, TAllocatorParams &&... allocator_params) /// NOLINT(performance-noexcept-swap)
    {
#ifndef NDEBUG
        this->unprotect();
        rhs.unprotect();
#endif

        /// Swap two PODArray objects, arr1 and arr2, that satisfy the following conditions:
        /// - The elements of arr1 are stored on stack.
        /// - The elements of arr2 are stored on heap.
        auto swap_stack_heap = [&](PODArray & arr1, PODArray & arr2)
        {
            size_t stack_size = arr1.size();
            size_t stack_allocated = arr1.allocated_bytes();

            size_t heap_size = arr2.size();
            size_t heap_allocated = arr2.allocated_bytes();

            /// Keep track of the stack content we have to copy.
            char * stack_c_start = arr1.c_start;

            /// arr1 takes ownership of the heap memory of arr2.
            arr1.c_start = arr2.c_start;
            arr1.c_end_of_storage = arr1.c_start + heap_allocated - arr2.pad_right - arr2.pad_left;
            arr1.c_end = arr1.c_start + PODArrayDetails::byte_size(heap_size, sizeof(T));

            /// Allocate stack space for arr2.
            arr2.alloc(stack_allocated, std::forward<TAllocatorParams>(allocator_params)...);
            /// Copy the stack content.
            memcpy(arr2.c_start, stack_c_start, PODArrayDetails::byte_size(stack_size, sizeof(T)));
            arr2.c_end = arr2.c_start + PODArrayDetails::byte_size(stack_size, sizeof(T));
        };

        auto do_move = [&](PODArray & src, PODArray & dest)
        {
            if (src.isAllocatedFromStack())
            {
                dest.dealloc();
                dest.alloc(src.allocated_bytes(), std::forward<TAllocatorParams>(allocator_params)...);
                memcpy(dest.c_start, src.c_start, PODArrayDetails::byte_size(src.size(), sizeof(T)));
                dest.c_end = dest.c_start + PODArrayDetails::byte_size(src.size(), sizeof(T));

                src.c_start = Base::null;
                src.c_end = Base::null;
                src.c_end_of_storage = Base::null;
            }
            else
            {
                std::swap(dest.c_start, src.c_start);
                std::swap(dest.c_end, src.c_end);
                std::swap(dest.c_end_of_storage, src.c_end_of_storage);
            }
        };

        if (!this->isInitialized() && !rhs.isInitialized())
        {
            return;
        }
        if (!this->isInitialized() && rhs.isInitialized())
        {
            do_move(rhs, *this);
            return;
        }
        if (this->isInitialized() && !rhs.isInitialized())
        {
            do_move(*this, rhs);
            return;
        }

        if (this->isAllocatedFromStack() && rhs.isAllocatedFromStack())
        {
            size_t min_size = std::min(this->size(), rhs.size());
            size_t max_size = std::max(this->size(), rhs.size());

            for (size_t i = 0; i < min_size; ++i)
                std::swap(this->operator[](i), rhs[i]);

            if (this->size() == max_size)
            {
                for (size_t i = min_size; i < max_size; ++i)
                    rhs[i] = this->operator[](i);
            }
            else
            {
                for (size_t i = min_size; i < max_size; ++i)
                    this->operator[](i) = rhs[i];
            }

            size_t lhs_size = this->size();
            size_t lhs_allocated = this->allocated_bytes();

            size_t rhs_size = rhs.size();
            size_t rhs_allocated = rhs.allocated_bytes();

            this->c_end_of_storage = this->c_start + rhs_allocated - Base::pad_right - Base::pad_left;
            rhs.c_end_of_storage = rhs.c_start + lhs_allocated - Base::pad_right - Base::pad_left;

            this->c_end = this->c_start + PODArrayDetails::byte_size(rhs_size, sizeof(T));
            rhs.c_end = rhs.c_start + PODArrayDetails::byte_size(lhs_size, sizeof(T));
        }
        else if (this->isAllocatedFromStack() && !rhs.isAllocatedFromStack())
        {
            swap_stack_heap(*this, rhs);
        }
        else if (!this->isAllocatedFromStack() && rhs.isAllocatedFromStack())
        {
            swap_stack_heap(rhs, *this);
        }
        else
        {
            std::swap(this->c_start, rhs.c_start);
            std::swap(this->c_end, rhs.c_end);
            std::swap(this->c_end_of_storage, rhs.c_end_of_storage);
        }
    }

    template <typename... TAllocatorParams>
    void assign(size_t n, const T & x, TAllocatorParams &&... allocator_params)
    {
        this->resize_exact(n, std::forward<TAllocatorParams>(allocator_params)...);
        std::fill(begin(), end(), x);
    }

    template <typename It1, typename It2, typename... TAllocatorParams>
    void assign(It1 from_begin, It2 from_end, TAllocatorParams &&... allocator_params)
    {
        static_assert(memcpy_can_be_used_for_assignment<std::decay_t<T>, std::decay_t<decltype(*from_begin)>>);
        this->assertNotIntersects(from_begin, from_end);

        size_t required_capacity = from_end - from_begin;
        if (required_capacity > this->capacity())
            this->reserve_exact(required_capacity, std::forward<TAllocatorParams>(allocator_params)...);

        size_t bytes_to_copy = PODArrayDetails::byte_size(required_capacity, sizeof(T));
        if (bytes_to_copy)
            memcpy(this->c_start, reinterpret_cast<const void *>(&*from_begin), bytes_to_copy);

        this->c_end = this->c_start + bytes_to_copy;
    }

    // ISO C++ has strict ambiguity rules, thus we cannot apply TAllocatorParams here.
    void assign(const PODArray & from)
    {
        assign(from.begin(), from.end());
    }

    void erase(const_iterator first, const_iterator last)
    {
        iterator first_no_const = const_cast<iterator>(first);
        iterator last_no_const = const_cast<iterator>(last);

        size_t items_to_move = end() - last;

        while (items_to_move != 0)
        {
            *first_no_const = *last_no_const;

            ++first_no_const;
            ++last_no_const;

            --items_to_move;
        }

        this->c_end = reinterpret_cast<char *>(first_no_const);
    }

    void erase(const_iterator pos)
    {
        this->erase(pos, pos + 1);
    }

    bool operator== (const PODArray & rhs) const
    {
        if (this->size() != rhs.size())
            return false;

        const_iterator lhs_it = begin();
        const_iterator rhs_it = rhs.begin();

        while (lhs_it != end())
        {
            if (*lhs_it != *rhs_it)
                return false;

            ++lhs_it;
            ++rhs_it;
        }

        return true;
    }

    bool operator!= (const PODArray & rhs) const
    {
        return !operator==(rhs);
    }
};

/// NOLINTEND(bugprone-sizeof-expression)

template <typename T, size_t initial_bytes, typename TAllocator, size_t pad_right_, size_t pad_left_>
void swap(PODArray<T, initial_bytes, TAllocator, pad_right_, pad_left_> & lhs, PODArray<T, initial_bytes, TAllocator, pad_right_, pad_left_> & rhs) /// NOLINT
{
    lhs.swap(rhs);
}

/// Prevent implicit template instantiation of PODArray for common numeric types

extern template class PODArray<UInt8, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;
extern template class PODArray<UInt16, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;
extern template class PODArray<UInt32, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;
extern template class PODArray<UInt64, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;

extern template class PODArray<Int8, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;
extern template class PODArray<Int16, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;
extern template class PODArray<Int32, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;
extern template class PODArray<Int64, 4096, Allocator<false>, PADDING_FOR_SIMD - 1, PADDING_FOR_SIMD>;

extern template class PODArray<UInt8, 4096, Allocator<false>, 0, 0>;
extern template class PODArray<UInt16, 4096, Allocator<false>, 0, 0>;
extern template class PODArray<UInt32, 4096, Allocator<false>, 0, 0>;
extern template class PODArray<UInt64, 4096, Allocator<false>, 0, 0>;

extern template class PODArray<Int8, 4096, Allocator<false>, 0, 0>;
extern template class PODArray<Int16, 4096, Allocator<false>, 0, 0>;
extern template class PODArray<Int32, 4096, Allocator<false>, 0, 0>;
extern template class PODArray<Int64, 4096, Allocator<false>, 0, 0>;

}
