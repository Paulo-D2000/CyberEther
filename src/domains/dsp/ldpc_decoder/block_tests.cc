#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>

#include "jetstream/domains/dsp/ldpc_decoder/block.hh"
#include "flowgraph_fixture.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Aligned LDPC demo is registered as an example",
          "[modules][dsp][ldpc_decoder][example]") {
    const auto examples = Registry::ListAvailableFlowgraphs("ldpc-aligned");
    REQUIRE(examples.size() == 1);
    REQUIRE(examples.front().title == "Aligned AR4JA LDPC Decoder");
    Parser::Map document;
    REQUIRE(Parser::YamlDecode(examples.front().content, document) == Result::SUCCESS);
    REQUIRE(document.contains("graph"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "LDPC decoder block wires llr input and bits output",
                 "[modules][dsp][ldpc_decoder][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("F32");
    sourceConfig["bufferSize"] = std::string("2048");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["llr"].requested("src", "signal");

    Blocks::LdpcDecoder config;
    config.iterations = 25;
    REQUIRE(flowgraph->blockCreate("ldpc", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("ldpc").state == Block::State::Created);

    const auto block = viewBlock("ldpc");
    const Tensor out = block.outputs.at("bits").tensor;
    REQUIRE(out.dtype() == DataType::U8);
    REQUIRE(out.elementSize() == 1);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 2560);

    REQUIRE(block.interfaceConfigs.size() == 1);
    REQUIRE(block.interfaceConfigs.at(0).name == "iterations");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("ldpc", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("iterations")) == config.iterations);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "LDPC decoder block is incomplete without required input",
                 "[modules][dsp][ldpc_decoder][block][validation]") {
    auto result = flowgraph->blockCreate("ldpc_incomplete", "ldpc_decoder",
                                         {}, {});
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("ldpc_incomplete").state == Block::State::Incomplete);
}
