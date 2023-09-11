//! Define the type responsible for rendering lines.

use bytemuck::{Pod, Zeroable};
use std::sync::mpsc;
use util::math::Point;

use crate::compute_shader::*;
use crate::shader::WgslLoader;

/// State required for rendering lines.
pub struct LineRenderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,

    compute_command_buf: wgpu::Buffer,
    compute_push_curve: ComputePipeline,

    rx: mpsc::Receiver<(Point, u32)>,
    num_indices: u64,
}

/// State required for requesting a line be rendered.
pub struct LineSender {
    tx: mpsc::Sender<(Point, u32)>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CurveCommand {
    pos: Point,
    curve_index: u32,
    _pad: u32,
}

pub fn line_renderer(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    surface: &wgpu::Surface,
) -> (LineSender, LineRenderer) {
    let (tx, rx) = mpsc::channel();
    let renderer = LineRenderer::with_channel(device, adapter, surface, rx);

    (LineSender { tx }, renderer)
}

impl LineRenderer {
    /// Given a device, command queue and a texture to draw to, render the lines.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let commands: Vec<CurveCommand> = self
            .rx
            .try_iter()
            .map(|(pos, curve_index)| CurveCommand {
                pos,
                curve_index,
                _pad: 0,
            })
            .collect();

        if !commands.is_empty() {
            queue.write_buffer(
                &self.compute_command_buf,
                0,
                bytemuck::cast_slice(commands.as_slice()),
            );

            // TODO: temporary hack, this will be replaced with compute-driven
            // indirect draws later
            if self.num_indices == 1 {
                self.num_indices = 0;
            } else {
                self.num_indices += 6 * commands.len() as u64;
            }

            self.compute_push_curve
                .run(&mut encoder, commands.len() as u32);
        }

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
            pass.set_vertex_buffer(0, self.vertex_buf.slice(8..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(2..(2 + self.num_indices as u32), 0, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Create a new LineRenderer given a device, adapter and surface.
    fn with_channel(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        surface: &wgpu::Surface,
        rx: mpsc::Receiver<(Point, u32)>,
    ) -> Self {
        // Create the index and vertex buffers
        let vertex_buf = create_buffer_with_intro(device, wgpu::BufferUsages::VERTEX, &[0]);
        let index_buf = create_buffer_with_intro(device, wgpu::BufferUsages::INDEX, &[0]);

        // Load the shaders
        let mut shader_loader = WgslLoader::new();
        shader_loader.add_common_source(include_str!("common.wgsl"));

        let render_shader = shader_loader.create_shader(device, include_str!("draw_line.wgsl"));

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
                module: &render_shader,
                entry_point: "vertex_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 2 * std::mem::size_of::<Point>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fragment_main",
                targets: &[Some(surface.get_capabilities(adapter).formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Describe and create the compute pipeline
        let push_curve_shader =
            shader_loader.create_shader(device, include_str!("push_curve.wgsl"));

        let compute_command_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16384,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let curve_buf = create_buffer_with_intro(device, wgpu::BufferUsages::COPY_DST, &[-1; 10]);

        let compute_push_curve = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Uniform(&compute_command_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
                ComputePipelineBuffer::Storage(&curve_buf),
            ],
            push_curve_shader,
            "push_curve_main",
        );

        Self {
            vertex_buf,
            index_buf,
            render_pipeline,
            compute_command_buf,
            compute_push_curve,
            rx,
            num_indices: 1,
        }
    }
}

impl LineSender {
    pub fn push_point(
        &self,
        point: Point,
        curve: u32,
    ) -> Result<(), mpsc::SendError<(Point, u32)>> {
        self.tx.send((point, curve))
    }
}

const DEFAULT_BUFFER_SIZE: u64 = 32760;

fn create_buffer_with_intro<T: bytemuck::Pod>(
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    intro_slice: &[T],
) -> wgpu::Buffer {
    let byte_slice: &[u8] = bytemuck::cast_slice(intro_slice);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: DEFAULT_BUFFER_SIZE,
        mapped_at_creation: true,
        usage: usage | wgpu::BufferUsages::STORAGE,
    });

    buffer
        .slice(0..byte_slice.len() as u64)
        .get_mapped_range_mut()
        .copy_from_slice(byte_slice);
    buffer.unmap();

    buffer
}
