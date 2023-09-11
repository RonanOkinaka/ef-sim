//! Common definitions for WGSL sources.

/// One vertex of a line segment.
struct Vertex {
    pos: vec2<f32>,
    ray: vec2<f32>,
}

/// Structure of the vertex buffer, which includes its size.
struct VertexBuffer {
    size: atomic<u32>,
    _pad: u32, // Explicit pad is easier to read
    data: array<Vertex>,
}

/// Structure of the index buffer, which includes its size.
struct IndexBuffer {
    size: atomic<u32>,
    _pad: u32,
    data: array<u32>,
}

/// Represents a curve as a linked list of points with line
/// segments between them.
struct Curve {
    head_index: i32,
    tail_index: i32,
}
