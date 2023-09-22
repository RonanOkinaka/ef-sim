//! Second pass of the circle renderer -- draw the billboards to the screen.

struct VertexIn {
    @location(0) pos: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
}


/// Circle texture.
@group(0) @binding(0) var circle_tex: texture_2d<f32>;

/// Circle sampler.
@group(0) @binding(1) var circle_samp: sampler;


// Vertex shader passes along texture coords as well.
@vertex
fn vertex_passthrough_tex(vertex: VertexIn) -> VertexOut {
    return VertexOut(
        vec4<f32>(vertex.pos.x, vertex.pos.y, 0.0, 1.0),
        vertex.tex_coords,
    );
}

// Fragment merely samples the texture.
@fragment
fn fragment_billboard(@location(1) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(circle_tex, circle_samp, tex_coords);
}
