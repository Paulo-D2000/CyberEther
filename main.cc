#define RENDER_DEBUG

#include "render/base.hpp"
#include "samurai/samurai.hpp"
#include "jetstream/fft/base.hpp"
#include "jetstream/lineplot/base.hpp"
#include "jetstream/waterfall/base.hpp"
#include "jetstream/histogram/base.hpp"

struct State {
    bool streaming = false;

    // Render
    std::shared_ptr<Render::Instance> render;

    // Samurai
    Samurai::ChannelId rx;
    std::shared_ptr<Samurai::Device> device;

    // Jetstream
    std::vector<std::complex<float>> stream;
    std::vector<std::shared_ptr<Jetstream::Module>> modules;
};

auto state = std::make_shared<State>();

void dsp_loop(std::shared_ptr<State> state) {
    while (state->streaming) {
        state->device->ReadStream(state->rx, state->stream.data(), state->stream.size(), 1000);
        JETSTREAM_ASSERT_SUCCESS(Jetstream::Compute(state->modules));
        JETSTREAM_ASSERT_SUCCESS(Jetstream::Barrier(state->modules));
    }
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    // Configure Render
    Render::Instance::Config renderCfg;
    renderCfg.width = 3130;
    renderCfg.height = 1140;
    renderCfg.resizable = true;
    renderCfg.enableImgui = true;
    renderCfg.enableVsync = true;
    renderCfg.title = "CyberEther";
    state->render = Render::Instantiate(Render::API::GLES, renderCfg);

    // Configure Samurai Radio
    state->device = std::make_shared<Samurai::Airspy::Device>();

    Samurai::Device::Config deviceConfig;
    deviceConfig.sampleRate = 10e6;
    state->device->Enable(deviceConfig);

    Samurai::Channel::Config channelConfig;
    channelConfig.mode = Samurai::Mode::RX;
    channelConfig.dataFmt = Samurai::Format::F32;
    state->device->EnableChannel(channelConfig, &state->rx);

    Samurai::Channel::State channelState;
    channelState.enableAGC = true;
    channelState.frequency = 96.9e6;
    state->device->UpdateChannel(state->rx, channelState);

    // Configure Jetstream Modules
    auto device = Jetstream::Locale::CPU;
    state->stream = std::vector<std::complex<float>>(8192*2);

    Jetstream::FFT::Config fftCfg;
    fftCfg.input0 = {Jetstream::Locale::CPU, state->stream};
    fftCfg.policy = {Jetstream::Policy::ASYNC, {}};
    auto fft = Jetstream::FFT::Instantiate(device, fftCfg);

    Jetstream::Lineplot::Config lptCfg;
    lptCfg.render = state->render;
    lptCfg.input0 = fft->output();
    lptCfg.policy = {Jetstream::Policy::ASYNC, {fft}};
    auto lpt = Jetstream::Lineplot::Instantiate(device, lptCfg);

    Jetstream::Waterfall::Config wtfCfg;
    wtfCfg.render = state->render;
    wtfCfg.input0 = fft->output();
    wtfCfg.policy = {Jetstream::Policy::ASYNC, {fft}};
    auto wtf = Jetstream::Waterfall::Instantiate(device, wtfCfg);

    // Add Jetstream modules to the execution pipeline.
    state->modules.push_back(fft);
    state->modules.push_back(lpt);
    state->modules.push_back(wtf);

    // Start Components
    state->streaming = true;
    state->render->create();
    state->device->StartStream();
    std::thread dsp(dsp_loop, state);

    while(state->render->keepRunning()) {
        state->render->start();

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        JETSTREAM_ASSERT_SUCCESS(Jetstream::Present(state->modules));

        ImGui::Begin("Lineplot");
        lptCfg.width = ImGui::GetContentRegionAvail().x;
        lptCfg.height = ImGui::GetContentRegionAvail().y;
        ImGui::Image((void*)(intptr_t)lpt->tex()->raw(), ImVec2(lpt->conf().width, lpt->conf().height));
        ImGui::End();

        ImGui::Begin("Waterfall");
        wtfCfg.width = ImGui::GetContentRegionAvail().x;
        wtfCfg.height = ImGui::GetContentRegionAvail().y;
        ImGui::Image((void*)(intptr_t)wtf->tex()->raw(), ImVec2(wtf->conf().width, wtf->conf().height));
        ImGui::End();

        ImGui::Begin("Control");
        ImGui::InputFloat("Frequency (Hz)", &channelState.frequency);
        if (ImGui::Button("Tune")) {
            state->device->UpdateChannel(state->rx, channelState);
        }
        ImGui::DragFloatRange2("dBFS Range", &fftCfg.min_db, &fftCfg.max_db,
             1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS");
        ImGui::End();

        ImGui::Begin("Samurai Info");
        float bufferUsageRatio = (float)state->device->BufferOccupancy(state->rx) /
            (float)state->device->BufferCapacity(state->rx);
        ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::Text("Buffer Usage");
        ImGui::End();

        state->render->end();
    }

    state->streaming = false;
    dsp.join();

    state->device->StopStream();
    state->render->destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
