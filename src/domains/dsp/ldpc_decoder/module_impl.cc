#include "module_impl.hh"

namespace Jetstream::Modules {

Result LdpcDecoderImpl::validate() {
    const auto& config = *candidate();
    if (config.iterations == 0 || config.iterations > 200) {
        JST_ERROR("[MODULE_LDPC_DECODER] Iterations must be between 1 and 200.");
        return Result::ERROR;
    }

    if (!inputs().contains("llr")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("llr").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_LDPC_DECODER] Unsupported data type '{}'.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    if (inputTensor.shape(inputTensor.rank() - 1) != kTxVarNodes) {
        JST_ERROR("[MODULE_LDPC_DECODER] Last input axis must be exactly "
                  "{} LLRs, got {}.",
                  kTxVarNodes, inputTensor.shape(inputTensor.rank() - 1));
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result LdpcDecoderImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("llr"));
    JST_CHECK(defineInterfaceOutput("bits"));

    return Result::SUCCESS;
}

Result LdpcDecoderImpl::create() {
    const Tensor& inputTensor = inputs().at("llr").tensor;

    input = inputTensor;
    sampleAxis = input.rank() - 1;
    laneCount = input.size() / input.shape(sampleAxis);

    auto outputShape = input.shape();
    outputShape[sampleAxis] = kVarNodes;
    JST_CHECK(output.create(input.device(), DataType::F32, outputShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["bits"].produced(name(), "bits", output);

    llr.assign(kVarNodes, 0.0f);
    checkToVar.assign(kNumEdges, 0.0f);
    varSum.assign(kVarNodes, 0.0f);

    return Result::SUCCESS;
}

Result LdpcDecoderImpl::destroy() {
    llr.clear();
    checkToVar.clear();
    varSum.clear();

    return Result::SUCCESS;
}

Result LdpcDecoderImpl::reconfigure() {
    const auto& config = *candidate();
    iterations = config.iterations;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
