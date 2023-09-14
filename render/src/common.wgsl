//! Common definitions for WGSL sources.

/// One vertex of a line segment.
struct Vertex {
    @location(0) pos: vec2<f32>,
    @location(1) ray: vec2<f32>,
    @location(2) alpha: f32,
    @location(3) next: i32,
}

/// Structure of the vertex buffer, which includes its size.
struct VertexBuffer {
    size: atomic<i32>,
    _pad: u32, // Explicit pad is easier to read
    data: array<Vertex>,
}

/// Structure of the index buffer, which includes its size.
struct IndexBuffer {
    size: atomic<i32>,
    data: array<i32>,
}

/// Represents a curve as a linked list of points with line
/// segments between them.
struct Curve {
    head_index: i32,
    tail_index: i32,
}

/// Flat free list for buffer pools.
struct FreeList {
    size: atomic<i32>,
    data: array<i32, $$FREE_LIST_LOGICAL_SIZE$$>,
}

/// Stores our higher-level logical constructs.
struct ComputeState {
    curves: array<Curve, $$CURVE_BUF_LOGICAL_SIZE$$>,
    vert_free: FreeList,
    indx_free: FreeList,
}
