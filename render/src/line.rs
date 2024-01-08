//! Define the type responsible for rendering lines.

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use rand::Rng;
use std::cmp::Ordering as CmpOrdering;
use std::collections::LinkedList;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use util::math::Point;

use crate::compute_shader::*;
use crate::render_util::*;
use crate::shader::WgslLoader;
use crate::update_queue::VecToWgpuBufHelper;

/// State required for rendering lines.
pub struct ParticleRenderer {
    // ParticleSubRenderer is very large and we really don't want to copy it
    sub_renderers: LinkedList<ParticleSubRenderer>,
    stencil_view: wgpu::TextureView,

    render_shader: wgpu::ShaderModule,
    particle_shader: wgpu::ShaderModule,
    pop_shader: wgpu::ShaderModule,

    charge_vec: Vec<Charge>,
    charge_buf: wgpu::Buffer,

    target_num_curves: Arc<AtomicU32>,
    current_num_curves: u32,

    start_time: std::time::Instant,
    last_frame: f32,
    // Gross hack required due to lack of atomic f32 support
    particle_lifetime_s: Arc<AtomicU32>,

    format: wgpu::TextureFormat,
    limits: ParticleBufferLimits,
}

/// State required for requesting a line be rendered.
#[derive(Clone)]
pub struct ParticleSender {
    target_num_curves: Arc<AtomicU32>,
    particle_lifetime_s: Arc<AtomicU32>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct ParticleBufferLimits {
    pub max_num_points: u32,
    pub max_num_curves: u32,
    pub max_num_charges: u32,

    pub default_buffer_size: u32,
    pub curve_buf_size_bytes: u32,

    pub workgroup_size_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Charge {
    pub pos: Point,
    pub charge: f32,
    pub _pad: u32,
}

struct ParticleSubRenderer {
    render_bundle: wgpu::RenderBundle,

    state_buf: wgpu::Buffer,
    param_buf: wgpu::Buffer,
    compute_particle: ComputePipeline,

    compute_pop_buf: wgpu::Buffer,
    compute_pop_curve: ComputePipeline,

    target_num_curves: u32,
    current_num_curves: u32,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Curve {
    head_index: i32,
    tail_index: i32,
    num_points: i32,
    lifetime: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rand_value: u32,
    tick_s: f32,
    particle_lifetime_s: f32,
}

// This is arbitrary, 16MB right now
const DEFAULT_BUFFER_SIZE: u32 = 16777216;

const PUSH_COMMAND_SIZE: u32 = 16;
const POP_COMMAND_SIZE: u32 = 4;
const CURVE_SIZE: u32 = 4 * 4;
const INDEX_SIZE: u32 = 4;
const VERTEX_SIZE: u32 = 6 * 4;
const QUAD_INDEX_SIZE: u32 = INDEX_SIZE * 6;
const INDIRECT_DRAW_SIZE: u32 = 5 * 4;
const CURVE_PRELUDE_SIZE: u32 = INDIRECT_DRAW_SIZE + 3 * 4;
const INDIRECT_DISPATCH_SIZE: u32 = 3 * 4;
const CHARGE_SIZE: u32 = 4 * 4;

const MAX_POINTS_PER_CURVE: u32 = 10;
const CHARGE_COLLISION_RADIUS: f32 = 0.1;

impl ParticleRenderer {
    pub fn get_charge_updater<'a>(
        &'a mut self,
        context: &'a RenderContext,
    ) -> VecToWgpuBufHelper<'a, Charge> {
        // TODO: This completely breaks encapsulation but I can't think of a better way
        // Just try not to misuse it :P
        VecToWgpuBufHelper {
            data_off: Some(CHARGE_SIZE.into()),
            size_off: Some(0),
            vec: &mut self.charge_vec,
            buf: &self.charge_buf,
            queue: context.queue,
        }
    }

    pub fn do_update_pass(&mut self, context: &RenderContext) {
        // Change the number of sub-renderers as necessary
        let target_num_curves = self.target_num_curves.load(Ordering::Relaxed);

        match target_num_curves.cmp(&self.current_num_curves) {
            CmpOrdering::Greater => {
                self.increase_num_particles(
                    target_num_curves - self.current_num_curves,
                    context.device,
                );
            }
            CmpOrdering::Less => {
                self.decrease_num_particles(self.current_num_curves - target_num_curves);
            }
            CmpOrdering::Equal => {}
        }
        self.current_num_curves = target_num_curves;

        // Generate the compute parameters
        let elapsed = self.start_time.elapsed();
        let now = elapsed.as_secs_f32();
        let params = Params {
            rand_value: 0,
            tick_s: now - self.last_frame,
            particle_lifetime_s: f32::from_bits(self.particle_lifetime_s.load(Ordering::Relaxed)),
        };
        self.last_frame = now;

        // Particle update pass
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            for part_renderer in self.sub_renderers.iter_mut() {
                part_renderer.do_particle_pass(
                    context.queue,
                    &mut pass,
                    Params {
                        rand_value: rand::thread_rng().gen(),
                        ..params
                    },
                );
            }
        }

        context.submit(encoder.finish()).unwrap();
    }

    pub fn do_pop_pass(&self, context: &RenderContext) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Pop points as necessary
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            for part_renderer in self.sub_renderers.iter() {
                part_renderer.do_pop_pass(&mut pass);
            }
        }

        context.submit(encoder.finish()).unwrap();
    }

    pub fn do_render_pass(&self, context: &RenderContext) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Finally, draw!
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: context.frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.stencil_view,
                    depth_ops: None,
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                }),
            });

            for part_renderer in self.sub_renderers.iter() {
                part_renderer.do_render_pass(&mut pass);
            }
        }

        context.submit(encoder.finish()).unwrap();
    }

    pub fn limits(&self) -> ParticleBufferLimits {
        self.limits
    }

    pub fn new(render_graph: &RenderGraph) -> Self {
        // Smaller between the default and max size for a storage buffer
        let default_buffer_size = render_graph
            .device
            .limits()
            .max_storage_buffer_binding_size
            .min(DEFAULT_BUFFER_SIZE);

        // Each free list gets one quarter (4), and each element is 4 bytes
        // Also, subtract out the size indicator
        let free_list_stack_size = (default_buffer_size / 4 / INDEX_SIZE) - 1;

        let max_num_points = free_list_stack_size // No more points than positions in free list
            .min(default_buffer_size / 2 / VERTEX_SIZE - 1) // One point is represented by 2 vertices
            .min(default_buffer_size / QUAD_INDEX_SIZE - 1); // One quad set is 6x 4-byte integers

        // Common shader data
        let mut shader_loader = WgslLoader::new();

        // Divide out the size of each command; subtract one for size indicator
        let max_push_per_frame = default_buffer_size / PUSH_COMMAND_SIZE - 1;
        let max_pops_per_frame =
            (default_buffer_size - INDIRECT_DISPATCH_SIZE) / POP_COMMAND_SIZE - 1;

        // The vertex + index free lists, and the curve structures, share space in one buffer
        // The curves get one half [without the indirect draw buffer]
        let curve_buf_size_bytes = (default_buffer_size) / 2 - CURVE_PRELUDE_SIZE;

        // And each one is 16 bytes
        let curve_buf_size = curve_buf_size_bytes / CURVE_SIZE;

        // Useful approximation to prevent buffer overruns
        let max_num_curves = curve_buf_size.min(max_num_points / (MAX_POINTS_PER_CURVE + 1));

        // Charge buffer size
        let charge_buf_size_bytes = render_graph
            .device
            .limits()
            .max_uniform_buffer_binding_size
            .min(16384);
        let max_num_charges = charge_buf_size_bytes / CHARGE_SIZE - 1;

        // Texture format
        let format = render_graph
            .surface
            .get_capabilities(&render_graph.adapter)
            .formats[0];

        // Load the shaders
        shader_loader.bind("PUSH_BUF_LOGICAL_SIZE", max_push_per_frame.to_string());
        shader_loader.bind("POP_BUF_LOGICAL_SIZE", max_pops_per_frame.to_string());
        shader_loader.bind("CURVE_BUF_LOGICAL_SIZE", curve_buf_size.to_string());
        shader_loader.bind("FREE_LIST_LOGICAL_SIZE", free_list_stack_size.to_string());
        shader_loader.bind("CHARGE_BUF_LOGICAL_SIZE", max_num_charges.to_string());

        shader_loader.bind("MAX_POINTS_PER_CURVE", MAX_POINTS_PER_CURVE.to_string());
        shader_loader.bind(
            "CHARGE_COLLISION_RADIUS",
            CHARGE_COLLISION_RADIUS.to_string(),
        );

        shader_loader.add_common_source(include_str!("common.wgsl"));
        shader_loader.bind(
            "INCLUDE_PUSH_COMMON_WGSL",
            include_str!("push_common.wgsl").to_owned(),
        );

        let workgroup_size_x = render_graph
            .device
            .limits()
            .max_compute_workgroup_size_x
            .min(256);
        shader_loader.bind("WORKGROUP_SIZE_X", workgroup_size_x.to_string());

        let render_shader =
            shader_loader.create_shader(&render_graph.device, include_str!("draw_line.wgsl"));
        let particle_shader =
            shader_loader.create_shader(&render_graph.device, include_str!("particle.wgsl"));
        let pop_shader =
            shader_loader.create_shader(&render_graph.device, include_str!("pop_curve.wgsl"));

        // Create the charge buffer
        let charge_buf = create_buffer_with_size(
            &render_graph.device,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            render_graph.device.limits().max_uniform_buffer_binding_size,
        );

        // Create the stencil view
        let stencil_buf = render_graph
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: render_graph
                    .surface
                    .get_current_texture()
                    .unwrap()
                    .texture
                    .size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Stencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

        let stencil_view = stencil_buf.create_view(&wgpu::TextureViewDescriptor::default());

        println!(
            "{:#?}",
            ParticleBufferLimits {
                max_num_points,
                max_num_curves,
                max_num_charges,
                default_buffer_size,
                curve_buf_size_bytes,
                workgroup_size_x,
            }
        );

        Self {
            sub_renderers: LinkedList::new(),
            stencil_view,
            charge_vec: Vec::new(),
            charge_buf,
            target_num_curves: Arc::new(AtomicU32::new(0)),
            current_num_curves: 0,
            start_time: std::time::Instant::now(),
            last_frame: 0.0,
            particle_lifetime_s: Arc::new(AtomicU32::new(10.0f32.to_bits())),
            render_shader,
            particle_shader,
            pop_shader,
            format,
            limits: ParticleBufferLimits {
                max_num_points,
                max_num_curves,
                max_num_charges,
                default_buffer_size,
                curve_buf_size_bytes,
                workgroup_size_x,
            },
        }
    }

    fn increase_num_particles(&mut self, mut new_curves_required: u32, device: &wgpu::Device) {
        // First, try to fill the last one
        if let Some(sub_renderer) = self.sub_renderers.back_mut() {
            let diff = (self.limits.max_num_curves - sub_renderer.current_num_curves)
                .min(new_curves_required);

            sub_renderer.target_num_curves += diff;
            new_curves_required -= diff;
        }

        // Now, start making new sub-renderers
        while new_curves_required > 0 {
            let diff = new_curves_required.min(self.limits.max_num_curves);
            new_curves_required -= diff;

            self.sub_renderers.push_back(ParticleSubRenderer::new(
                device,
                self.format,
                diff,
                self.limits,
                &self.render_shader,
                &self.particle_shader,
                &self.pop_shader,
                &self.charge_buf,
            ));
        }
    }

    fn decrease_num_particles(&mut self, mut num_curves_deleted: u32) {
        // Pop back to front
        let mut sub_render_iter = self.sub_renderers.iter_mut().rev();

        while num_curves_deleted > 0 {
            let sub_renderer = sub_render_iter.next().expect("Miscounted active");

            let diff = num_curves_deleted.min(sub_renderer.current_num_curves);

            sub_renderer.target_num_curves -= diff;
            num_curves_deleted -= diff;
        }

        // TODO: Write a compaction shader to reduce the number of drawn indices and
        // active curves when their numbers are reduced
    }
}

impl ParticleSubRenderer {
    fn do_particle_pass<'a>(
        &'a mut self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::ComputePass<'a>,
        params: Params,
    ) {
        // Reset the pop buffer and dispatch data
        queue.write_buffer(
            &self.compute_pop_buf,
            0,
            cast_slice::<u32, u8>(&[
                1, // Pop workgroup x [It would be nice if this could be 0, but I'm not sure if that's allowed]
                1, // Pop workgroup y
                1, // Pop workgroup z
                0, // # points to pop
            ]),
        );

        // Write the compute parameters
        queue.write_buffer(&self.param_buf, 0, cast_slice(&[params]));

        // Update the curve buffer size as necessary
        if self.target_num_curves != self.current_num_curves {
            self.current_num_curves = self.current_num_curves.max(self.target_num_curves);
            queue.write_buffer(
                &self.state_buf,
                INDIRECT_DRAW_SIZE.into(),
                cast_slice(&[self.current_num_curves, self.target_num_curves]),
            );
        }

        // Compute
        self.compute_particle
            .run_shared(pass, self.current_num_curves);
    }

    fn do_pop_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        self.compute_pop_curve
            .run_indirect_shared(pass, &self.compute_pop_buf, 0);
    }

    fn do_render_pass<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.execute_bundles([&self.render_bundle]);
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        target_num_curves: u32,
        limits: ParticleBufferLimits,
        render_shader: &wgpu::ShaderModule,
        particle_shader: &wgpu::ShaderModule,
        pop_shader: &wgpu::ShaderModule,
        charge_buf: &wgpu::Buffer,
    ) -> Self {
        // Create the index and vertex buffers
        let vertex_buf = create_buffer_with_size(
            device,
            wgpu::BufferUsages::VERTEX,
            limits.default_buffer_size,
        );
        let index_buf = create_buffer_with_size(
            device,
            wgpu::BufferUsages::INDEX,
            limits.default_buffer_size,
        );

        // Create the state buffer
        let state_buf = create_buffer_with_size_mapped(
            device,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            limits.default_buffer_size,
            true,
        );
        {
            let mut curve_slice = state_buf
                .slice(..(CURVE_PRELUDE_SIZE as u64))
                .get_mapped_range_mut();
            cast_slice_mut(&mut curve_slice).copy_from_slice(bytemuck::cast_slice::<u32, u8>(&[
                0, // # indices
                1, // # instances (1, always)
                0, // Index buffer offset
                0, // Vertex buffer offset
                0, // Instance offset
                0, // Dispatch size
                0, // Curve buffer target size
                0, // Curve buffer current size
            ]));

            let mut curve_slice = state_buf
                .slice((CURVE_PRELUDE_SIZE as u64)..(limits.curve_buf_size_bytes as u64))
                .get_mapped_range_mut();
            cast_slice_mut(&mut curve_slice).fill(Curve {
                head_index: -1,
                tail_index: -1,
                num_points: 0,
                lifetime: 0.0,
            });
        }
        state_buf.unmap();

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
                module: render_shader,
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
                module: render_shader,
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
                    ..format.into()
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Stencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Equal,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Invert,
                    },
                    back: wgpu::StencilFaceState::IGNORE,
                    read_mask: 0xFF,
                    write_mask: 0xFF,
                },
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create the render bundle
        let mut render_bundle_encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: None,
                color_formats: &[Some(format)],
                depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                    format: wgpu::TextureFormat::Stencil8,
                    depth_read_only: true,
                    stencil_read_only: false,
                }),
                sample_count: wgpu::MultisampleState::default().count,
                multiview: None,
            });

        render_bundle_encoder.set_pipeline(&render_pipeline);
        render_bundle_encoder.set_vertex_buffer(0, vertex_buf.slice(8..));
        render_bundle_encoder.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        render_bundle_encoder.draw_indexed_indirect(&state_buf, 0);

        let render_bundle =
            render_bundle_encoder.finish(&wgpu::RenderBundleDescriptor { label: None });

        // Describe and create the pop pipeline
        let compute_pop_buf = create_buffer_with_size(
            device,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            limits.default_buffer_size,
        );

        let compute_pop_curve = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::StorageReadOnly(&compute_pop_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
                ComputePipelineBuffer::Storage(&state_buf),
            ],
            pop_shader,
            "pop_curve_main",
            limits.workgroup_size_x,
        );

        // Create the particle compute pipeline
        let param_buf = create_buffer_with_size(
            device,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            std::mem::size_of::<Params>() as u32,
        );

        let compute_particle = ComputePipeline::new(
            device,
            &[
                ComputePipelineBuffer::Storage(&state_buf),
                ComputePipelineBuffer::Uniform(charge_buf),
                ComputePipelineBuffer::Storage(&compute_pop_buf),
                ComputePipelineBuffer::Storage(&vertex_buf),
                ComputePipelineBuffer::Storage(&index_buf),
                ComputePipelineBuffer::Uniform(&param_buf),
            ],
            particle_shader,
            "particle_main",
            limits.workgroup_size_x,
        );

        Self {
            render_bundle,
            state_buf,
            param_buf,
            compute_particle,
            compute_pop_buf,
            compute_pop_curve,
            target_num_curves,
            current_num_curves: 0,
        }
    }
}

impl ParticleSender {
    pub fn from_renderer(particle_renderer: &ParticleRenderer) -> Self {
        Self {
            target_num_curves: particle_renderer.target_num_curves.clone(),
            particle_lifetime_s: particle_renderer.particle_lifetime_s.clone(),
        }
    }

    pub fn set_num_curves(&self, num_curves: u32) {
        self.target_num_curves.store(num_curves, Ordering::Relaxed);
    }

    pub fn set_particle_lifetime(&self, lifetime_s: f32) {
        self.particle_lifetime_s
            .store(lifetime_s.to_bits(), Ordering::Relaxed);
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
