@vertex
fn vertex_main(@location(0) pos: vec2<f32>, @location(1) ray: vec2<f32>) -> @builtin(position) vec4<f32> {
    let scaled = ray * 0.01; // TODO: This should be a parameter!
    return vec4<f32>(pos.x + scaled.x, pos.y + scaled.y, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
