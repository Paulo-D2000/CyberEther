#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/dsp/ldpc_decoder/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"
#include "test_data/ldpc_golden.hh"

using namespace Jetstream;

namespace {

constexpr U64 kFrameLlrs = 2048;

// All-zero codeword under channel LLR +3 with `flipCount` deterministic
// hard-decision flips (LLR -3): the decoder must recover every flip.
Tensor MakeFlippedLlrs(const std::vector<U64>& shape, const U64 flipCount) {
    Tensor input(DeviceType::CPU, DataType::F32, shape);
    for (U64 i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = 3.0f;
    }
    if (flipCount > 0) {
        const U64 laneLlrs = shape.back();
        for (U64 flip = 0; flip < flipCount; ++flip) {
            const U64 index = (flip * 97 + 13) % laneLlrs;
            for (U64 lane = 0; lane < input.size() / laneLlrs; ++lane) {
                input.at<F32>(lane * laneLlrs + index) = -3.0f;
            }
        }
    }
    return input;
}

void RequireLdpcValidationError(const Registry::ModuleRegistration& impl,
                                const Modules::LdpcDecoder& config,
                                const std::vector<U64>& shape,
                                const DataType dtype = DataType::F32) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);

    TensorMap inputs;
    inputs["llr"].requested("test", "llr");
    inputs["llr"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("ldpc_decoder", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("LDPC decoder - recovers flipped bits from all-zero codeword",
          "[modules][dsp][ldpc_decoder]") {
    const auto implementations = Registry::ListAvailableModules("ldpc_decoder");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("ldpc_decoder", impl.device, impl.runtime, impl.provider);
            Modules::LdpcDecoder config;
            config.iterations = 50;
            ctx.setConfig(config);

            Tensor input = MakeFlippedLlrs({kFrameLlrs}, 20);
            ctx.setInput("llr", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("bits");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 2560);

            for (U64 i = 0; i < out.size(); ++i) {
                REQUIRE(out.at<F32>(i) == 0.0f);
            }
        }
    }
}

TEST_CASE("LDPC decoder - decodes lanes independently",
          "[modules][dsp][ldpc_decoder]") {
    const auto implementations = Registry::ListAvailableModules("ldpc_decoder");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("ldpc_decoder", impl.device, impl.runtime, impl.provider);
            Modules::LdpcDecoder config;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32, {2, kFrameLlrs});
            for (U64 i = 0; i < kFrameLlrs; ++i) {
                input.at<F32>(0, i) = 3.0f;                          // lane 0: clean
                input.at<F32>(1, i) =
                    ((i * 31 + 7) % 41 == 0) ? -3.0f : 3.0f;         // lane 1: ~50 flips
            }
            ctx.setInput("llr", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("bits");
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 2560);
            for (U64 lane = 0; lane < 2; ++lane) {
                for (U64 i = 0; i < 2560; ++i) {
                    REQUIRE(out.at<F32>(lane, i) == 0.0f);
                }
            }
        }
    }
}

TEST_CASE("LDPC decoder - golden vectors from the validated pipeline",
          "[modules][dsp][ldpc_decoder][golden]") {
    const auto implementations = Registry::ListAvailableModules("ldpc_decoder");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("ldpc_decoder", impl.device, impl.runtime, impl.provider);
            Modules::LdpcDecoder config;
            config.iterations = 50;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32,
                         {static_cast<U64>(ldpc_golden::kFrames),
                          static_cast<U64>(ldpc_golden::kFrameLlrs)});
            for (I64 f = 0; f < ldpc_golden::kFrames; ++f) {
                for (I64 i = 0; i < ldpc_golden::kFrameLlrs; ++i) {
                    input.at<F32>(f, i) = ldpc_golden::kLlrs[f][i];
                }
            }
            ctx.setInput("llr", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("bits");
            REQUIRE(out.shape(0) == static_cast<U64>(ldpc_golden::kFrames));
            REQUIRE(out.shape(1) == static_cast<U64>(ldpc_golden::kFrameBits));

            U64 mismatches = 0;
            for (I64 f = 0; f < ldpc_golden::kFrames; ++f) {
                for (I64 i = 0; i < ldpc_golden::kFrameBits; ++i) {
                    const F32 expected =
                        static_cast<F32>(ldpc_golden::kCodewords[f][i]);
                    if (out.at<F32>(f, i) != expected) {
                        ++mismatches;
                    }
                }
            }
            REQUIRE(mismatches == 0);
        }
    }
}

TEST_CASE("LDPC decoder - rejects invalid configuration and input",
          "[modules][dsp][ldpc_decoder][validation]") {
    const auto implementations = Registry::ListAvailableModules("ldpc_decoder");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::LdpcDecoder config;

            SECTION("Zero iterations") {
                config.iterations = 0;
                RequireLdpcValidationError(impl, config, {kFrameLlrs});
            }
            SECTION("Wrong frame length") {
                RequireLdpcValidationError(impl, config, {1000});
            }
            SECTION("Wrong data type") {
                RequireLdpcValidationError(impl, config, {kFrameLlrs},
                                           DataType::CF32);
            }
        }
    }
}
