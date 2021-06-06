#ifndef RENDER_GLES_DRAW_H
#define RENDER_GLES_DRAW_H

#include "render/gles/instance.hpp"

namespace Render {

class GLES::Draw : public Render::Draw {
public:
    Draw(Config& cfg, GLES& i) : Render::Draw(cfg), inst(i) {};

protected:
    GLES& inst;

    std::shared_ptr<GLES::Vertex> buffer;

    Result create();
    Result destroy();
    Result draw();

    friend class GLES::Program;
};

} // namespace Render

#endif