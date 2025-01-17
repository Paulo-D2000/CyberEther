#ifndef JETSTREAM_MODULES_AUDIO_HH
#define JETSTREAM_MODULES_AUDIO_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

#include "jetstream/tools/miniaudio.h"

namespace Jetstream {

#define JST_AUDIO_CPU(MACRO) \
    MACRO(Audio, CPU, F32) \

template<Device D, typename T = F32>
class Audio : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        F32 inSampleRate = 48e3;
        F32 outSampleRate = 48e3;

        JST_SERDES(inSampleRate, outSampleRate);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();
    Result destroy();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
    ma_device_config deviceConfig;
    ma_device deviceCtx;
    ma_resampler_config resamplerConfig;
    ma_resampler resamplerCtx;

    Memory::CircularBuffer<F32> buffer;

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE
JST_AUDIO_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
