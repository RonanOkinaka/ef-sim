//! Define the type responsible for rendering lines.

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use rand::Rng;
use std::cmp::Ordering as CmpOrdering;
use std::collections::LinkedList;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use util::math::Point;

use crate::compute_shader::*;
use crate::render_util::Renderer;
use crate::shader::WgslLoader;
use crate::update_queue::{UpdateQueue, UpdateResult};

/// State required for rendering lines.
pub struct ParticleRenderer {
    // ParticleSubRenderer is very large and we really don't want to copy it
    sub_renderers: LinkedList<ParticleSubRenderer>,

    render_shader: wgpu::ShaderModule,
    particle_shader: wgpu::ShaderModule,
    pop_shader: wgpu::ShaderModule,

    charge_updates: UpdateQueue<Charge>,
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
    charge_updates: UpdateQueue<Charge>,
    target_num_curves: Arc<AtomicU32>,
    particle_lifetime_s: Arc<AtomicU32>,
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

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
struct ParticleBufferLimits {
    max_num_points: u32,
    max_num_curves: u32,
    max_num_charges: u32,

    default_buffer_size: u32,
    curve_buf_size_bytes: u32,

    workgroup_size_x: u32,
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
struct Charge {
    pos: Point,
    charge: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rand_value: u32,
    tick_s: f32,
    particle_lifetime_s: f32,
}

pub fn particle_renderer(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    surface: &wgpu::Surface,
) -> (ParticleSender, ParticleRenderer) {
    let renderer = ParticleRenderer::new(device, adapter, surface);

    (
        ParticleSender {
            charge_updates: renderer.charge_updates.clone(),
            target_num_curves: renderer.target_num_curves.clone(),
            particle_lifetime_s: renderer.particle_lifetime_s.clone(),
        },
        renderer,
    )
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

impl Renderer for ParticleRenderer {
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        // Update charge buffer
        self.update_charge_buffer(queue);

        // Change the number of sub-renderers as necessary
        let target_num_curves = self.target_num_curves.load(Ordering::Relaxed);

        match target_num_curves.cmp(&self.current_num_curves) {
            CmpOrdering::Greater => {
                self.increase_num_particles(target_num_curves - self.current_num_curves, device);
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

        // Prepare our command encoder
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Order matters here! First, update our particles
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            for part_renderer in self.sub_renderers.iter_mut() {
                part_renderer.do_particle_pass(
                    queue,
                    &mut pass,
                    Params {
                        rand_value: rand::thread_rng().gen(),
                        ..params
                    },
                );
            }
        }

        // Next, pop points as necessary
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

            for part_renderer in self.sub_renderers.iter_mut() {
                part_renderer.do_pop_pass(&mut pass);
            }
        }

        // Finally, draw!
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

            for part_renderer in self.sub_renderers.iter_mut() {
                part_renderer.do_render_pass(&mut pass);
            }
        }

        encoder.finish()
    }
}

impl ParticleRenderer {
    fn new(device: &wgpu::Device, adapter: &wgpu::Adapter, surface: &wgpu::Surface) -> Self {
        // Smaller between the default and max size for a storage buffer
        let default_buffer_size = device
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
        let charge_buf_size_bytes = device.limits().max_uniform_buffer_binding_size.min(16384);
        let max_num_charges = charge_buf_size_bytes / CHARGE_SIZE - 1;

        // Texture format
        let format = surface.get_capabilities(adapter).formats[0];

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

        let workgroup_size_x = 256.min(device.limits().max_compute_workgroup_size_x);
        shader_loader.bind("WORKGROUP_SIZE_X", workgroup_size_x.to_string());

        let render_shader = shader_loader.create_shader(device, include_str!("draw_line.wgsl"));
        let particle_shader = shader_loader.create_shader(device, include_str!("particle.wgsl"));
        let pop_shader = shader_loader.create_shader(device, include_str!("pop_curve.wgsl"));

        // Create the charge buffer
        let charge_buf = create_buffer_with_size(
            device,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            device.limits().max_uniform_buffer_binding_size,
        );

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
            charge_updates: UpdateQueue::with_limit(max_num_charges as usize),
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

    fn update_charge_buffer(&mut self, queue: &wgpu::Queue) {
        match self.charge_updates.apply_updates(&mut self.charge_vec) {
            UpdateResult::Range(min, max) => {
                // Write the charge buffer
                // This buffer shouldn't be very large, so copying the whole
                // thing isn't a huge deal (TODO: segmented write algorithm?)
                queue.write_buffer(
                    &self.charge_buf,
                    (min as u64 + 1) * CHARGE_SIZE as u64,
                    cast_slice(&self.charge_vec[min..max]),
                );
            }
            UpdateResult::SizeOnly(..) => {} // Only update the size
            UpdateResult::Same => return,    // Early return if nothing happened
        };

        // Update the charge counts
        queue.write_buffer(
            &self.charge_buf,
            0,
            cast_slice(&[self.charge_vec.len() as u32]),
        );
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

    fn do_pop_pass<'a>(&'a mut self, pass: &mut wgpu::ComputePass<'a>) {
        self.compute_pop_curve
            .run_indirect_shared(pass, &self.compute_pop_buf, 0);
    }

    fn do_render_pass<'a>(&'a mut self, pass: &mut wgpu::RenderPass<'a>) {
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create the render bundle
        let mut render_bundle_encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: None,
                color_formats: &[Some(format)],
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
    pub fn push_charge(&self, pos: Point, charge: f32) -> Result<u32, (Point, f32)> {
        let charge_data = Charge {
            pos,
            charge,
            _pad: 0,
        };

        match self.charge_updates.push(charge_data) {
            Ok(index) => Ok(index as u32),
            Err(..) => Err((pos, charge)),
        }
    }

    pub fn pop_charge(&self, index: u32) -> Result<(), u32> {
        match self.charge_updates.pop(index as usize) {
            Ok(()) => Ok(()),
            Err(..) => Err(index),
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
