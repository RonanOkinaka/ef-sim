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

    compute_push_buf: wgpu::Buffer,
    compute_push_curve: ComputePipeline,

    compute_pop_buf: wgpu::Buffer,
    compute_pop_curve: ComputePipeline,

    push_rx: mpsc::Receiver<(Point, u32)>,
    pop_rx: mpsc::Receiver<u32>,
    num_indices: u64,
}

/// State required for requesting a line be rendered.
pub struct LineSender {
    push_tx: mpsc::Sender<(Point, u32)>,
    pop_tx: mpsc::Sender<u32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushCommand {
    pos: Point,
    curve_index: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PopCommand {
    curve_index: u32,
    _pad: [u32; 3],
}

pub fn line_renderer(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    surface: &wgpu::Surface,
) -> (LineSender, LineRenderer) {
    let (push_tx, push_rx) = mpsc::channel();
    let (pop_tx, pop_rx) = mpsc::channel();
    let renderer = LineRenderer::with_channel(device, adapter, surface, push_rx, pop_rx);

    (LineSender { push_tx, pop_tx }, renderer)
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

        let push_commands: Vec<PushCommand> = self
            .push_rx
            .try_iter()
            .map(|(pos, curve_index)| PushCommand {
                pos,
                curve_index,
                _pad: 0,
            })
            .collect();

        if !push_commands.is_empty() {
            queue.write_buffer(
                &self.compute_push_buf,
                0,
                bytemuck::cast_slice(push_commands.as_slice()),
            );

            // TODO: temporary hack, this will be replaced with compute-driven
            // indirect draws later
            if self.num_indices == 1 {
                self.num_indices = 0;
            } else {
                self.num_indices += 6 * push_commands.len() as u64;
            }

            self.compute_push_curve
                .run(&mut encoder, push_commands.len() as u32);
        }

        let pop_commands: Vec<PopCommand> = self
            .pop_rx
            .try_iter()
            .map(|curve_index| PopCommand {
                curve_index,
                _pad: [0; 3],
            })
            .collect();

        if !pop_commands.is_empty() {
            queue.write_buffer(
                &self.compute_pop_buf,
                0,
                bytemuck::cast_slice(pop_commands.as_slice()),
            );

            self.compute_pop_curve
                .run(&mut encoder, pop_commands.len() as u32);
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
        push_rx: mpsc::Receiver<(Point, u32)>,
        pop_rx: mpsc::Receiver<u32>,
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
                    array_stride: 4 * 6, // 6 units of 4 bytes each (see below)
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Float32,
                        3 => Sint32,
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fragment_main",
                targets: &[Some(wgpu::ColorTargetState {
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    ..surface.get_capabilities(adapter).formats[0].into()
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Common compute shader data
        let command_buf_size = device.limits().max_uniform_buffer_binding_size.min(32768) as u64;
        shader_loader.bind("COMMAND_BUF_SIZE", (command_buf_size / 16).to_string()); // Divide by size

        // Describe and create the push pipeline
        let push_curve_shader =
            shader_loader.create_shader(device, include_str!("push_curve.wgsl"));

        let compute_push_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: command_buf_size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let curve_buf = create_buffer_with_intro(device, wgpu::BufferUsages::COPY_DST, &[-1; 10]);

        let compute_push_curve = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Uniform(&compute_push_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
                ComputePipelineBuffer::Storage(&curve_buf),
            ],
            push_curve_shader,
            "push_curve_main",
        );

        // Describe and create the pop pipeline
        let compute_pop_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: command_buf_size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let compute_pop_curve = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Uniform(&compute_pop_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&curve_buf),
            ],
            shader_loader.create_shader(device, include_str!("pop_curve.wgsl")),
            "pop_curve_main",
        );

        Self {
            vertex_buf,
            index_buf,
            render_pipeline,
            compute_push_buf,
            compute_push_curve,
            compute_pop_buf,
            compute_pop_curve,
            push_rx,
            pop_rx,
            num_indices: 1,
        }
    }
}

// TODO: This should be a memory-mapped shared buffer
impl LineSender {
    pub fn push_point(
        &self,
        point: Point,
        curve: u32,
    ) -> Result<(), mpsc::SendError<(Point, u32)>> {
        self.push_tx.send((point, curve))
    }

    pub fn pop_point(&self, curve: u32) -> Result<(), mpsc::SendError<u32>> {
        self.pop_tx.send(curve)
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
