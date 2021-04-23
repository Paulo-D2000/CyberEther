#include "render/gles/vertex.hpp"

namespace Render {

Result GLES::Vertex::create() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    int i = 0;
    for (auto& buffer : cfg.buffers) {
        uint usage = GL_STATIC_DRAW;
        switch (buffer.usage) {
            case Vertex::Buffer::Usage::Dynamic:
                usage = GL_DYNAMIC_DRAW;
                break;
            case Vertex::Buffer::Usage::Stream:
                usage = GL_STREAM_DRAW;
                break;
            case Vertex::Buffer::Usage::Static:
                usage = GL_STATIC_DRAW;
                break;
        }

        glGenBuffers(1, &buffer.index);
        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        glBufferData(GL_ARRAY_BUFFER, buffer.size * sizeof(float), buffer.data, usage);
        glVertexAttribPointer(i, buffer.stride, GL_FLOAT, GL_FALSE, buffer.stride * sizeof(float), 0);
        glEnableVertexAttribArray(i++);
    }

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cfg.indices.size() * sizeof(uint), cfg.indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::destroy() {
    for (auto& buffer : cfg.buffers) {
        glDeleteBuffers(1, &buffer.index);
    }
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::draw() {
    glBindVertexArray(vao);
    switch (cfg.mode) {
        case Vertex::Mode::Triangles:
            glDrawElements(GL_TRIANGLES, cfg.indices.size(), GL_UNSIGNED_INT, 0);
            break;
        case Vertex::Mode::Lines:
            glDrawElements(GL_LINES, cfg.indices.size(), GL_UNSIGNED_INT, 0);
            break;
        case Vertex::Mode::Points:
            glDrawElements(GL_POINTS, cfg.indices.size(), GL_UNSIGNED_INT, 0);
            break;
        case Vertex::Mode::LineLoop:
            glDrawElements(GL_LINE_LOOP, cfg.indices.size(), GL_UNSIGNED_INT, 0);
            break;
    }
    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Vertex::update() {
    glBindVertexArray(vao);
    for (auto& buffer : cfg.buffers) {
        glBindBuffer(GL_ARRAY_BUFFER, buffer.index);
        glBufferSubData(GL_ARRAY_BUFFER, 0, buffer.size * sizeof(float), buffer.data);
    }
    glBindVertexArray(0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render