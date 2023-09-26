//! Ultra-simplistic utility for dealing with WGSL shader sources.
//! TODO: It would be nice to add some tools for automatically
//! determining the size, alignment and offset of shader structures
//! but we can live without it for now.

use regex::{Regex, Replacer};
use std::collections::HashMap;

/// Structure which handles WGSL loading state.
pub struct WgslLoader {
    prelude_src: String,
    replacer: MappedReplacer,
    regex: Regex,
}

struct MappedReplacer {
    replace_map: HashMap<String, String>,
}

impl WgslLoader {
    /// Create a default WgslLoader.
    pub fn new() -> Self {
        let regex = Regex::new(concat!(
            "[$]{2}",        // $$
            "[a-zA-Z0-9_]+", // SOME_PATTERN_0
            "[$]{2}"         // $$
        ));

        Self {
            prelude_src: String::new(),
            replacer: MappedReplacer {
                replace_map: HashMap::new(),
            },
            regex: regex.unwrap(),
        }
    }

    /// "Bind" one string to another -- replacing e.g. $$SOME_CONSTANT$$ with
    /// its bound string.
    /// Supports alphabetic (upper or lower), numeric characters and underscores
    /// only.
    /// This is not recursive, and including a section of $$...$$ in the mapped
    /// string will have no effect.
    pub fn bind(&mut self, replace: &str, with: String) {
        self.replacer
            .replace_map
            .insert(format!("$${}$$", replace), with);
    }

    /// Push source lines verbatim to the beginning of all following files,
    /// inserting one newline at the end.
    pub fn add_common_source(&mut self, src: &str) {
        self.prelude_src.push_str(src);
        self.prelude_src.push('\n');
    }

    /// Creates a shader module equivalent to the provided string preceded by
    /// all prior arguments to add_common_source(), and with all $$CONSTANTS$$
    /// replaced if a match is found.
    pub fn create_shader(&mut self, device: &wgpu::Device, src: &str) -> wgpu::ShaderModule {
        // No reason that this should be mut, but by_ref() is mutable...
        let prelude_iter = self
            .regex
            .replace_all(&self.prelude_src, self.replacer.by_ref());
        let src_iter = self.regex.replace_all(src, self.replacer.by_ref());

        let final_src: String = prelude_iter.chars().chain(src_iter.chars()).collect();

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&final_src)),
        })
    }
}

impl Replacer for MappedReplacer {
    fn replace_append(&mut self, caps: &regex::Captures<'_>, dst: &mut String) {
        let full_cap = caps[0].to_owned();

        // Replace if we find a match; otherwise, do nothing
        dst.push_str(self.replace_map.get(&full_cap).unwrap_or(&full_cap));
    }
}
