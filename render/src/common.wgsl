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

/// Structure of the index buffer, which _doesn't_ include its size
/// (which lives in IndirectIndexedDraw now)
struct IndexBuffer {
    data: array<i32>,
}

/// Represents a curve as a linked list of points with line
/// segments between them.
struct Curve {
    head_index: i32,
    tail_index: i32,
    num_points: i32,
}

/// Buffer for all curves.
struct CurveBuffer {
    size: u32,
    data: array<Curve, $$CURVE_BUF_LOGICAL_SIZE$$>,
}

/// Flat free list for buffer pools.
struct FreeList {
    size: atomic<i32>,
    data: array<i32, $$FREE_LIST_LOGICAL_SIZE$$>,
}

/// Indirect draw data.
struct IndirectIndexedDraw {
    num_indices: atomic<i32>,

    // We won't actually write any of the below!
    num_insts: u32,
    indx_offset: u32,
    vert_offset: u32,
    inst_offset: u32,
}

/// Stores our higher-level logical constructs.
struct ComputeState {
    draw: IndirectIndexedDraw,
    curves: CurveBuffer,
    vert_free: FreeList,
    indx_free: FreeList,
}

/// Data for a compute shader dispatch.
struct IndirectComputeDispatch {
    workgroups_x: atomic<u32>,
    workgroups_y: u32, // For our purposes, y and z will always be 1
    workgroups_z: u32,
}

/// Data required for popping from a curve.
struct PopCommand {
    curve_index: u32,
}

/// Buffer for accumulating curve-pop commands.
struct PopCommandBuffer {
    dispatch: IndirectComputeDispatch,
    size: atomic<u32>,
    data: array<PopCommand, $$POP_BUF_LOGICAL_SIZE$$>,
}
