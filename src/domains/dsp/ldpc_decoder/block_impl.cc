#include <jetstream/domains/dsp/ldpc_decoder/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/dsp/ldpc_decoder/module.hh>

namespace Jetstream::Blocks {

struct LdpcDecoderImpl : public Block::Impl,
                         public DynamicConfig<Blocks::LdpcDecoder> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::LdpcDecoder> moduleConfig =
        std::make_shared<Modules::LdpcDecoder>();
};

Result LdpcDecoderImpl::configure() {
    moduleConfig->iterations = iterations;

    return Result::SUCCESS;
}

Result LdpcDecoderImpl::define() {
    JST_CHECK(defineInterfaceInput("llr",
                                   "Input",
                                   "Channel LLRs (2048 per frame)."));
    JST_CHECK(defineInterfaceOutput("bits",
                                    "Output",
                                    "Decoded hard decisions (2560 per frame)."));

    JST_CHECK(defineInterfaceConfig("iterations",
                                    "Iterations",
                                    "Maximum min-sum iterations.",
                                    {{"type", "uint"}, {"unit", "iterations"}}));

    return Result::SUCCESS;
}

Result LdpcDecoderImpl::create() {
    JST_CHECK(moduleCreate("ldpc_decoder", moduleConfig, {
        {"llr", inputs().at("llr")}
    }));
    JST_CHECK(moduleExposeOutput("bits", {"ldpc_decoder", "bits"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(LdpcDecoderImpl, {"ldpc_decoder"});

}  // namespace Jetstream::Blocks
