#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

using TI = tcb::span<std::complex<float>>;
using TO = tcb::span<float>;

struct Config {
    float max_db = 0.0;
    float min_db = -200.0;
    Data<TI> input0;
    Module::Execution policy;
};

class Generic : public Module {
public:
    explicit Generic(Config& c)
        : Module(c.policy),
          cfg(c),
          in(c.input0) {
    }
    virtual ~Generic() = default;

    Config conf() const {
        return cfg;
    }

    Data<TO> output() const {
        return out;
    }

protected:
    Config& cfg;
    Data<TI> in;
    Data<TO> out;
};

} // namespace Jetstream::FFT

#endif