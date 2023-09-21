//! Define the type responsible for rendering lines.

use bytemuck::{cast_slice_mut, Pod, Zeroable};
use std::collections::HashMap;
use std::sync::mpsc;
use util::math::Point;

use crate::compute_shader::*;
use crate::shader::WgslLoader;

/// State required for rendering lines.
#[allow(dead_code)]
pub struct LineRenderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    render_bundle: wgpu::RenderBundle,

    charge_buf: wgpu::Buffer,
    compute_particle: ComputePipeline,

    compute_push_buf: wgpu::Buffer,
    compute_push_curve: ComputePipeline,

    compute_pop_buf: wgpu::Buffer,
    compute_pop_curve: ComputePipeline,

    push_rx: mpsc::Receiver<(Point, u32)>,
    pop_rx: mpsc::Receiver<u32>,

    max_push_per_frame: u32,
    max_pops_per_frame: u32,
    max_num_points: u32,
    max_num_curves: u32,
    max_num_charges: u32,
}

/// State required for requesting a line be rendered.
#[derive(Clone)]
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

// This is arbitrary, 16MB right now
const DEFAULT_BUFFER_SIZE: u32 = 16777216;

const PUSH_COMMAND_SIZE: u32 = 16;
const PUSH_COMMAND_ALIGN: u32 = 8;
const POP_COMMAND_SIZE: u32 = 4;
const POP_COMMAND_ALIGN: u32 = 4;
const CURVE_SIZE: u32 = 8;
const INDEX_SIZE: u32 = 4;
const VERTEX_SIZE: u32 = 6 * 4;
const QUAD_INDEX_SIZE: u32 = INDEX_SIZE * 6;
const INDIRECT_DRAW_SIZE: u32 = 6 * 4;
const CHARGE_SIZE: u32 = 4 * 4;

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

        let push_commands_iter = self.push_rx.try_iter().map(|(pos, curve_index)| {
            (
                curve_index,
                PushCommand {
                    pos,
                    curve_index,
                    _pad: 0,
                },
            )
        });

        let push_commands = partition_curve_updates(push_commands_iter, self.max_push_per_frame);

        if push_commands.is_empty() {
            // Reset the pop buffer
            queue.write_buffer(&self.compute_pop_buf, 0, &[0; POP_COMMAND_ALIGN as usize]);

            self.compute_particle.run(&mut encoder, 1);
        }

        for push_pass in push_commands {
            let len = push_pass.len() as u32;

            queue.write_buffer(&self.compute_push_buf, 0, bytemuck::cast_slice(&[len]));
            queue.write_buffer(
                &self.compute_push_buf,
                PUSH_COMMAND_ALIGN.into(),
                bytemuck::cast_slice(push_pass.as_slice()),
            );

            self.compute_push_curve.run(&mut encoder, len);
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

            pass.execute_bundles([&self.render_bundle]);
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
        // Smaller between the default and max size for a storage buffer
        let default_buffer_size = device
            .limits()
            .max_storage_buffer_binding_size
            .min(DEFAULT_BUFFER_SIZE);

        // Create the index and vertex buffers
        let vertex_buf =
            create_buffer_with_size(device, wgpu::BufferUsages::VERTEX, default_buffer_size);
        let index_buf =
            create_buffer_with_size(device, wgpu::BufferUsages::INDEX, default_buffer_size);

        // Common shader data
        let mut shader_loader = WgslLoader::new();

        // Divide out the size of each command; subtract one for size indicator
        let max_push_per_frame = default_buffer_size / PUSH_COMMAND_SIZE - 1;
        let max_pops_per_frame = default_buffer_size / POP_COMMAND_SIZE - 1;

        // The vertex + index free lists, and the curve structures, share space in one buffer
        // The curves get one half [without the indirect draw buffer]
        let curve_buf_size_bytes = (default_buffer_size) / 2 - INDIRECT_DRAW_SIZE;

        // And each one is 8 bytes
        let max_num_curves = curve_buf_size_bytes / CURVE_SIZE;

        // Each free list gets one quarter (4), and each element is 4 bytes
        // Also, subtract out the size indicator
        let free_list_stack_size = (default_buffer_size / 4 / INDEX_SIZE) - 1;

        // Charge buffer size
        let charge_buf_size_bytes = device.limits().max_uniform_buffer_binding_size.min(16384);
        let max_num_charges = charge_buf_size_bytes / CHARGE_SIZE;

        shader_loader.bind("PUSH_BUF_LOGICAL_SIZE", max_push_per_frame.to_string());
        shader_loader.bind("POP_BUF_LOGICAL_SIZE", max_pops_per_frame.to_string());
        shader_loader.bind("CURVE_BUF_LOGICAL_SIZE", max_num_curves.to_string());
        shader_loader.bind("FREE_LIST_LOGICAL_SIZE", free_list_stack_size.to_string());
        shader_loader.bind("CHARGE_BUF_LOGICAL_SIZE", max_num_charges.to_string());

        shader_loader.add_common_source(include_str!("common.wgsl"));
        shader_loader.bind(
            "INCLUDE_PUSH_COMMON_WGSL",
            include_str!("push_common.wgsl").to_owned(),
        );

        // Create the state buffer
        let state_buf = create_buffer_with_size_mapped(
            device,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            default_buffer_size,
            true,
        );
        {
            let mut curve_slice = state_buf
                .slice(..(INDIRECT_DRAW_SIZE as u64))
                .get_mapped_range_mut();
            cast_slice_mut(&mut curve_slice).copy_from_slice(bytemuck::cast_slice::<u32, u8>(&[
                0, // # indices
                1, // # instances (1, always)
                0, // Index buffer offset
                0, // Vertex buffer offset
                0, // Instance offset
                1, // Curve buffer size
            ]));

            let mut curve_slice = state_buf
                .slice((INDIRECT_DRAW_SIZE as u64)..(curve_buf_size_bytes as u64))
                .get_mapped_range_mut();
            cast_slice_mut(&mut curve_slice).fill(-1);
        }
        state_buf.unmap();

        // Render shader
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
                    array_stride: VERTEX_SIZE as u64,
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

        // Create the render bundle
        let mut render_bundle_encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: None,
                color_formats: &[Some(surface.get_capabilities(adapter).formats[0])],
                depth_stencil: None,
                sample_count: wgpu::MultisampleState::default().count,
                multiview: None,
            });

        render_bundle_encoder.set_pipeline(&render_pipeline);
        render_bundle_encoder.set_vertex_buffer(0, vertex_buf.slice(8..));
        render_bundle_encoder.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        render_bundle_encoder.draw_indexed_indirect(&state_buf, 0);

        let render_bundle =
            render_bundle_encoder.finish(&wgpu::RenderBundleDescriptor { label: None });

        // General compute data
        let workgroup_size_x = 256;
        shader_loader.bind("WORKGROUP_SIZE_X", workgroup_size_x.to_string());

        // Describe and create the push pipeline
        let compute_push_buf =
            create_buffer_with_size(device, wgpu::BufferUsages::COPY_DST, default_buffer_size);

        let push_curve_shader =
            shader_loader.create_shader(device, include_str!("push_curve.wgsl"));

        let compute_push_curve = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Storage(&compute_push_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
                ComputePipelineBuffer::Storage(&state_buf),
            ],
            push_curve_shader,
            "push_curve_main",
            workgroup_size_x,
        );

        // Describe and create the pop pipeline
        let compute_pop_buf =
            create_buffer_with_size(device, wgpu::BufferUsages::COPY_DST, default_buffer_size);

        let compute_pop_curve = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Storage(&compute_pop_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
                ComputePipelineBuffer::Storage(&state_buf),
            ],
            shader_loader.create_shader(device, include_str!("pop_curve.wgsl")),
            "pop_curve_main",
            workgroup_size_x,
        );

        let max_num_points = free_list_stack_size // No more points than positions in free list
            .min(vertex_buf.size() as u32 / 2 / VERTEX_SIZE - 1) // One point is represented by 2 vertices
            .min(index_buf.size() as u32 / QUAD_INDEX_SIZE - 1); // One quad set is 6x 4-byte integers

        println!(
            "max push: {}\nmax pops: {}\nmax points: {}\nmax curves: {}",
            max_push_per_frame, max_pops_per_frame, max_num_points, max_num_curves
        );

        // Create the particle compute pipeline
        let charge_buf = create_buffer_with_size(
            device,
            wgpu::BufferUsages::UNIFORM,
            device.limits().max_uniform_buffer_binding_size,
        );

        let compute_particle_shader =
            shader_loader.create_shader(device, include_str!("particle.wgsl"));

        let compute_particle = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Storage(&state_buf),
                ComputePipelineBuffer::Uniform(&charge_buf),
                ComputePipelineBuffer::Storage(&compute_pop_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
            ],
            compute_particle_shader,
            "particle_main",
            workgroup_size_x,
        );

        Self {
            vertex_buf,
            index_buf,
            render_pipeline,
            render_bundle,
            charge_buf,
            compute_particle,
            compute_push_buf,
            compute_push_curve,
            compute_pop_buf,
            compute_pop_curve,
            push_rx,
            pop_rx,
            max_push_per_frame,
            max_pops_per_frame,
            max_num_points,
            max_num_curves,
            max_num_charges,
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

fn create_buffer_with_size_mapped(
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    size: u32,
    mapped_at_creation: bool,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        mapped_at_creation,
        usage: usage | wgpu::BufferUsages::STORAGE,
    })
}

fn create_buffer_with_size(
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    size: u32,
) -> wgpu::Buffer {
    create_buffer_with_size_mapped(device, usage, size, false)
}

/// In order of arrival, partition curve updates such that
/// one curve is not more than once per pass, and the number
/// of updates does not surpass the command buffer size.
fn partition_curve_updates<T, I>(iter: I, limit: u32) -> Vec<Vec<T>>
where
    I: Iterator<Item = (u32, T)>,
{
    let limit = limit as usize;
    let mut ret = Vec::<Vec<_>>::new();
    let mut gen_map = HashMap::<u32, i32>::new();
    let mut first_not_full = 0;
    let mut len = 0;

    for (curve, value) in iter {
        let gen = gen_map.entry(curve).or_insert(-1);
        *gen = first_not_full.max(*gen + 1);

        let gen = *gen as usize;
        if gen >= len {
            ret.push(Vec::new());
            len += 1;
        }

        ret[gen].push(value);
        if ret[gen].len() == limit {
            first_not_full += 1;
        }
    }

    ret
}
