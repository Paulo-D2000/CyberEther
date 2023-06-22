#include "jetstream/modules/constellation.hh"
#include "shaders/constellation_shaders.hh"

namespace Jetstream {

template<Device D, typename T>
Constellation<D, T>::Constellation(const Config& config,
                                   const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Constellation module.");
    JST_CHECK_THROW(Module::initInput(input.buffer));
}

template<Device D, typename T>
void Constellation<D, T>::summary() const {
    JST_INFO("     Window Size: [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Constellation<D, T>::createPresent(Render::Window& window) {
    Render::Buffer::Config fillScreenVerticesConf;
    fillScreenVerticesConf.buffer = &Render::Extras::FillScreenVertices;
    fillScreenVerticesConf.elementByteSize = sizeof(float);
    fillScreenVerticesConf.size = 12;
    fillScreenVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window.build(fillScreenVerticesBuffer, fillScreenVerticesConf));

    Render::Buffer::Config fillScreenTextureVerticesConf;
    fillScreenTextureVerticesConf.buffer = &Render::Extras::FillScreenTextureVertices;
    fillScreenTextureVerticesConf.elementByteSize = sizeof(float);
    fillScreenTextureVerticesConf.size = 8;
    fillScreenTextureVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window.build(fillScreenTextureVerticesBuffer, fillScreenTextureVerticesConf));

    Render::Buffer::Config fillScreenIndicesConf;
    fillScreenIndicesConf.buffer = &Render::Extras::FillScreenIndices;
    fillScreenIndicesConf.elementByteSize = sizeof(uint32_t);
    fillScreenIndicesConf.size = 6;
    fillScreenIndicesConf.target = Render::Buffer::Target::VERTEX_INDICES;
    JST_CHECK(window.build(fillScreenIndicesBuffer, fillScreenIndicesConf));

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = {
        {fillScreenVerticesBuffer, 3},
        {fillScreenTextureVerticesBuffer, 2},
    };
    vertexCfg.indices = fillScreenIndicesBuffer;
    JST_CHECK(window.build(vertex, vertexCfg));

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Mode::TRIANGLES;
    JST_CHECK(window.build(drawVertex, drawVertexCfg));

    Render::Texture::Config bufferCfg;
    bufferCfg.buffer = (U8*)(timeSamples.data());
    bufferCfg.size = {timeSamples.shape()[0], timeSamples.shape()[1]};
    bufferCfg.dfmt = Render::Texture::DataFormat::F32;
    bufferCfg.pfmt = Render::Texture::PixelFormat::RED;
    bufferCfg.ptype = Render::Texture::PixelType::F32;
    JST_CHECK(window.build(binTexture, bufferCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    JST_CHECK(window.build(lutTexture, lutTextureCfg));

    // TODO: This could use unified memory.
    Render::Buffer::Config uniformCfg;
    uniformCfg.buffer = &shaderUniforms;
    uniformCfg.elementByteSize = sizeof(shaderUniforms);
    uniformCfg.size = 1;
    uniformCfg.target = Render::Buffer::Target::STORAGE;
    JST_CHECK(window.build(uniformBuffer, uniformCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = {
       {Device::Metal,  {signal_msl_vert_shader, signal_msl_frag_shader}},
       {Device::Vulkan, {signal_spv_vert_shader, signal_spv_frag_shader}},
    };
    programCfg.draw = drawVertex;
    programCfg.textures = {binTexture, lutTexture};
    programCfg.buffers = {
        {uniformBuffer, Render::Program::Target::VERTEX |
                        Render::Program::Target::FRAGMENT},
    };
    JST_CHECK(window.build(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window.build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK(window.build(surface, surfaceCfg));
    JST_CHECK(window.bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::present(Render::Window&) {
    binTexture->fill();

    shaderUniforms.width = timeSamples.shape()[0];
    shaderUniforms.height = timeSamples.shape()[1];
    shaderUniforms.zoom = 1.0;
    shaderUniforms.offset = 0.0;
    uniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
const Render::Size2D<U64>& Constellation<D, T>::viewSize(const Render::Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Constellation size changed from [{}, {}] to [{}, {}].", 
                config.viewSize.width, 
                config.viewSize.height, 
                viewSize.width, 
                viewSize.height);

        this->config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Constellation<D, T>::getTexture() {
    return *texture;
};

template<Device D, typename T>
Result Constellation<D, T>::Factory(std::unordered_map<std::string, std::any>& configMap,
                                    std::unordered_map<std::string, std::any>& inputMap,
                                    std::unordered_map<std::string, std::any>&,
                                    std::shared_ptr<Constellation<D, T>>& module) {
    using Module = Constellation<D, T>;

    Module::Config config{};

    JST_CHECK(Module::BindVariable(configMap, "viewSize", config.viewSize));

    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "buffer", input.buffer));

    module = std::make_shared<Module>(config, input);

    return Result::SUCCESS;
}

}  // namespace Jetstream
