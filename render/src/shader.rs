//! Ultra-simplistic utility for dealing with WGSL shader sources.
//! TODO: It would be nice to add some tools for automatically
//! determining the size, alignment and offset of shader structures
//! but we can live without it for now.

/// Structure which handles WGSL loading state.
pub struct WgslLoader {
    prelude_src: String,
}

impl WgslLoader {
    /// Create a default WgslLoader.
    pub fn new() -> Self {
        Self {
            prelude_src: String::new(),
        }
    }

    /// Push source lines verbatim to the beginning of all following files,
    /// inserting one newline at the end.
    pub fn add_common_source(&mut self, src: &str) {
        self.prelude_src.push_str(src);
        self.prelude_src.push('\n');
    }

    /// Creates a shader module equivalent to the provided string preceded by
    /// all prior arguments to add_common_source().
    pub fn create_shader(&self, device: &wgpu::Device, src: &str) -> wgpu::ShaderModule {
        let mut final_src = self.prelude_src.clone();
        final_src.push_str(src);

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&final_src)),
        })
    }
}
