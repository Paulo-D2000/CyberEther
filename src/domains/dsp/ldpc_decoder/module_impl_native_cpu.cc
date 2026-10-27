#include <algorithm>
#include <cmath>
#include <limits>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr F32 kMaxF32 = std::numeric_limits<F32>::max();

// Flooded min-sum with the 2-min trick and early parity termination.
// Mirrors the validated reference decoder (LDPC-Project decoder_minsum_csr)
// but with per-edge message storage instead of a dense MxN matrix.
void DecodeLane(const F32* inputLane,
                U8* outputLane,
                const U64 sampleStrideIn,
                const U64 sampleStrideOut,
                F32* llr,
                F32* checkToVar,
                F32* varSum,
                const U64 iterations) {
    // Channel LLRs: 2048 transmitted symbols, punctured tail at zero.
    for (I64 v = 0; v < kTxVarNodes; ++v) {
        llr[v] = inputLane[v * sampleStrideIn];
    }
    std::fill(llr + kTxVarNodes, llr + kVarNodes, 0.0f);

    for (I64 e = 0; e < kNumEdges; ++e) {
        checkToVar[e] = llr[kCheckVar[e]];
    }

    for (U64 iter = 0; iter < iterations; ++iter) {
        // Check-node pass: exclusive sign-min per edge.
        for (I64 r = 0; r < kCheckNodes; ++r) {
            const I64 begin = kCheckOffsets[r];
            const I64 end = kCheckOffsets[r + 1];

            F32 min1 = kMaxF32;
            F32 min2 = kMaxF32;
            F32 sign = 1.0f;
            for (I64 e = begin; e < end; ++e) {
                const F32 v = checkToVar[e];
                if (v < 0.0f) {
                    sign = -sign;
                }
                const F32 magnitude = std::fabs(v);
                if (magnitude < min1) {
                    min2 = min1;
                    min1 = magnitude;
                } else if (magnitude < min2) {
                    min2 = magnitude;
                }
            }

            for (I64 e = begin; e < end; ++e) {
                const F32 v = checkToVar[e];
                const F32 exclusiveSign = (v < 0.0f) ? -sign : sign;
                const F32 exclusiveMin = (std::fabs(v) == min1) ? min2 : min1;
                checkToVar[e] = exclusiveSign * exclusiveMin;
            }
        }

        // Variable-node pass: total belief, hard decision, extrinsic update.
        std::copy(llr, llr + kVarNodes, varSum);
        for (I64 e = 0; e < kNumEdges; ++e) {
            varSum[kCheckVar[e]] += checkToVar[e];
        }
        for (I64 v = 0; v < kVarNodes; ++v) {
            outputLane[v * sampleStrideOut] = (varSum[v] < 0.0f) ? 1 : 0;
        }
        for (I64 e = 0; e < kNumEdges; ++e) {
            const I64 v = kCheckVar[e];
            checkToVar[e] = varSum[v] - checkToVar[e];
        }

        // Early termination: all parity checks satisfied.
        bool valid = true;
        for (I64 r = 0; r < kCheckNodes && valid; ++r) {
            I32 parity = 0;
            for (I64 e = kCheckOffsets[r]; e < kCheckOffsets[r + 1]; ++e) {
                parity ^= static_cast<I32>(outputLane[kCheckVar[e] * sampleStrideOut]);
            }
            if (parity != 0) {
                valid = false;
            }
        }
        if (valid) {
            break;
        }
    }
}

}  // namespace

struct LdpcDecoderImplNativeCpu : public LdpcDecoderImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernel();
};

Result LdpcDecoderImplNativeCpu::validate() {
    return LdpcDecoderImpl::validate();
}

Result LdpcDecoderImplNativeCpu::create() {
    JST_CHECK(LdpcDecoderImpl::create());

    return Result::SUCCESS;
}

Result LdpcDecoderImplNativeCpu::computeSubmit() {
    return kernel();
}

Result LdpcDecoderImplNativeCpu::kernel() {
    const F32* in = input.data<F32>();
    U8* out = output.data<U8>();
    const U64 sampleCount = input.shape(sampleAxis);
    const U64 inputSampleStride = input.stride(sampleAxis);
    const U64 outputSampleStride = output.stride(sampleAxis);

    for (U64 lane = 0; lane < laneCount; ++lane) {
        U64 coordinates = lane;
        U64 inputLaneOffset = 0;
        U64 outputLaneOffset = 0;
        for (Index axis = input.rank(); axis-- > 0;) {
            if (axis == sampleAxis) {
                continue;
            }
            const U64 coordinate = coordinates % input.shape(axis);
            coordinates /= input.shape(axis);
            inputLaneOffset += coordinate * input.stride(axis);
            outputLaneOffset += coordinate * output.stride(axis);
        }
        (void)sampleCount;

        DecodeLane(in + inputLaneOffset,
                   out + outputLaneOffset,
                   inputSampleStride,
                   outputSampleStride,
                   llr.data(),
                   checkToVar.data(),
                   varSum.data(),
                   iterations);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(LdpcDecoderImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
