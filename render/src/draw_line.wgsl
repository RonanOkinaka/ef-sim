//! Render pipeline shaders for curves.

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
}

@vertex
fn vertex_main(vertex: Vertex) -> VertexOut {
    let scaled = vertex.ray * 0.005; // TODO: This should be a parameter!
    return VertexOut(
        vec4<f32>(vertex.pos.x + scaled.x, vertex.pos.y + scaled.y, 0.0, 1.0),
        vec4<f32>(1.0, 0.0, 0.0, vertex.alpha)
    );
}

@fragment
fn fragment_main(data: VertexOut) -> @location(0) vec4<f32> {
    return data.color;
}
