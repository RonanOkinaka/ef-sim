//! Define the type responsible for rendering lines.

use bytemuck::bytes_of;
use util::math::Point;
use std::mem::size_of;

/// State required for rendering lines.
pub struct LineRenderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,

    render_pipeline: wgpu::RenderPipeline,
}

impl LineRenderer {
    /// Create a new LineRenderer given a device, adapter and surface.
    pub fn new(device: &wgpu::Device, adapter: &wgpu::Adapter, surface: &wgpu::Surface) -> Self {
        // Create the index and vertex buffers
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4 * size_of::<Point>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 6 * size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        // Load the shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Borrowed(include_str!("draw_line.wgsl")),
            ),
        });

        // Describe and create the render pipeline
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vertex_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Point>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fragment_main",
                targets: &[Some(surface.get_capabilities(adapter).formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self { vertex_buf, index_buf, render_pipeline }
    }

    /// Given a device, command queue and a texture to draw to, render the lines.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
    ) {
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None }
        );

        // This is hard-coded for now -- just get something on the screen!
        let vertex_data = [
            Point(-0.5, 0.5), Point(0.5, 0.5),
            Point(-0.5, -0.5), Point(0.5, -0.5),
        ];
        let index_data = [
            0, 2, 3,
            0, 3, 1u32,
        ];

        // This does not need to be copied every time; just a placeholder for me
        queue.write_buffer(&self.vertex_buf, 0, bytes_of(&vertex_data));
        queue.write_buffer(&self.index_buf, 0, bytes_of(&index_data));

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None, 
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..6, 0, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}
