#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <complex>
#include <future>
#include <iostream>
#include <vector>

#include "jetstream_config.hpp"
#include "tools/span.hpp"

#ifndef JETSTREAM_ASSERT_SUCCESS
#define JETSTREAM_ASSERT_SUCCESS(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception (" <<  magic_enum::enum_name(result) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

namespace Jetstream {

enum Result {
    SUCCESS = 0,
    ERROR = 1,
    UNKNOWN = 2,
    ERROR_FUTURE_INVALID,
};

enum class Policy : uint8_t {
    HYBRID  = 0,
    ASYNC   = 1,
    SYNC    = 2,
};

enum class Locale : uint8_t {
    NONE    = 0 << 0,
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

template<typename T>
struct Data {
    Locale location;
    T buf;
};

} // namespace Jetstream

#endif