//! First pass of the circle renderer -- generate the billboards for the second pass.

// Vertex shader does nothing.
@vertex
fn vertex_passthrough(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos.x, pos.y, 0.0, 1.0);
}

// Pixel-accurate circle with the fragment shader.
@fragment
fn fragment_pixel_perfect(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let center_len = f32($$BILLBOARD_TEXTURE_DIM$$) / 2.0;
    let screen_pos = vec2<f32>(frag_pos.x, frag_pos.y) - vec2<f32>(center_len, center_len);

    if (length(screen_pos) < center_len) {
        return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
