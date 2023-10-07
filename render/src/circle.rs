//! Renderer for drawing solid-color, billboarded circles.

use bytemuck::cast_slice;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use util::math::Point;
use wgpu::util::DeviceExt;

use crate::render_util::*;
use crate::shader::WgslLoader;

/// Renderer responsible for drawing circle billboards.
pub struct CircleRenderer {
    num_verts: AtomicU32,
    max_num_verts: u32,
    vertices: wgpu::Buffer,

    billboard: wgpu::TextureView,
    billboard_verts: wgpu::Buffer,
    billboard_pipeline: wgpu::RenderPipeline,

    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
}

/// Channel by which one may request a circle be drawn.
#[derive(Clone)]
pub struct CircleSender {
    circle_tx: mpsc::Sender<(Point, f32)>,
}

pub fn circle_renderer(
    render_graph: &mut RenderGraph,
    max_num_billboards: u32,
) -> (CircleSender, usize) {
    let (circle_tx, circle_rx) = mpsc::channel();

    let renderer = Arc::new(CircleRenderer::new(render_graph, max_num_billboards));

    let updater = renderer.clone();
    let update_handle = render_graph.add_pass(Box::new(move |context| {
        updater.update_with_channel(context, &circle_rx);
    }));

    let render_handle = render_graph.add_pass(Box::new(move |context| {
        renderer.render(context);
    }));

    render_graph.set_sequence(update_handle, RenderOrder::RunsBefore, render_handle);

    (CircleSender { circle_tx }, render_handle)
}

const FLOATS_PER_VERTEX: u64 = 4;
const VERTEX_SIZE: u64 = 4 * FLOATS_PER_VERTEX;
const VERTS_PER_QUAD: u64 = 6;
const FLOATS_PER_QUAD: u64 = VERTS_PER_QUAD * 4;

impl CircleRenderer {
    fn update_with_channel(
        &self,
        context: &RenderContext,
        circle_rx: &mpsc::Receiver<(Point, f32)>,
    ) {
        let current_num_verts = self.num_verts.load(Ordering::Acquire);
        let num_quads_allowed = (self.max_num_verts - current_num_verts) / VERTS_PER_QUAD as u32;

        let verts: Vec<_> = circle_rx
            .try_iter()
            .map(|(pos, radius)| Self::circle_to_slice(pos, radius))
            .take(num_quads_allowed as usize)
            .flatten()
            .collect();

        if !verts.is_empty() {
            // Write the new vertex data
            context.queue.write_buffer(
                &self.vertices,
                VERTEX_SIZE * current_num_verts as u64,
                cast_slice(verts.as_slice()),
            );

            // If this is our first quad, create the billboard texture as well
            if current_num_verts == 0 {
                let mut encoder = context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                self.create_billboard(&mut encoder);

                context.queue.submit([encoder.finish()]);
            }

            // Update our count
            self.num_verts.fetch_add(
                verts.len() as u32 / FLOATS_PER_VERTEX as u32,
                Ordering::Release,
            );
        }

        context.submit_empty().unwrap();
    }

    fn render(&self, context: &RenderContext) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: context.frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..self.num_verts.load(Ordering::Acquire), 0..1);
        }

        context.submit(encoder.finish()).unwrap();
    }

    fn new(render_graph: &mut RenderGraph, max_num_billboards: u32) -> Self {
        // Create the billboard generation pipeline
        let best_format = render_graph
            .surface
            .get_capabilities(&render_graph.adapter)
            .formats[0];

        let billboard_texture = render_graph
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 512, // TODO: This should be dynamic
                    height: 512,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: best_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

        let billboard = billboard_texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let mut shader_loader = WgslLoader::new();
        shader_loader.bind(
            "BILLBOARD_TEXTURE_DIM",
            billboard_texture.width().to_string(),
        );
        let shader = shader_loader.create_shader(
            &render_graph.device,
            include_str!("draw_circle_texture.wgsl"),
        );

        let billboard_layout =
            render_graph
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        let billboard_pipeline =
            render_graph
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&billboard_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vertex_passthrough",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                        }],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fragment_pixel_perfect",
                        targets: &[Some(best_format.into())],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                });

        let billboard_verts =
            render_graph
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: cast_slice::<f32, u8>(&[
                        1.0, 1.0, // 0
                        -1.0, 1.0, // 1
                        -1.0, -1.0, // 2
                        -1.0, -1.0, // 2
                        1.0, -1.0, // 3
                        1.0, 1.0, // 0
                    ]),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        // Create the standard render pipeline
        let vertices = render_graph.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: VERTEX_SIZE * 6 * max_num_billboards as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader = shader_loader.create_shader(
            &render_graph.device,
            include_str!("draw_circle_billboards.wgsl"),
        );

        let render_bind_group_layout =
            render_graph
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let render_layout =
            render_graph
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&render_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline =
            render_graph
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&render_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vertex_passthrough_tex",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: VERTEX_SIZE,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![
                                0 => Float32x2,
                                1 => Float32x2,
                            ],
                        }],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fragment_billboard",
                        targets: &[Some(wgpu::ColorTargetState {
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::OVER,
                            }),
                            ..render_graph
                                .surface
                                .get_capabilities(&render_graph.adapter)
                                .formats[0]
                                .into()
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                });

        let sampler = render_graph
            .device
            .create_sampler(&wgpu::SamplerDescriptor {
                label: None,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        let render_bind_group = render_graph
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&billboard),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

        Self {
            vertices,
            num_verts: AtomicU32::new(0),
            max_num_verts: max_num_billboards * VERTS_PER_QUAD as u32,
            billboard,
            billboard_verts,
            billboard_pipeline,
            render_pipeline,
            render_bind_group,
        }
    }

    #[rustfmt::skip]
    fn circle_to_slice(pos: Point, width: f32) -> [f32; FLOATS_PER_QUAD as usize] {
        let right = pos.0 + width;
        let left = pos.0 - width;
        let top = pos.1 + width;
        let bottom = pos.1 - width;

        [
            right, top, 1.0, 0.0,
            left, top, 0.0, 0.0,
            left, bottom, 0.0, 1.0,
            left, bottom, 0.0, 1.0,
            right, bottom, 1.0, 1.0,
            right, top, 1.0, 0.0,
        ]
    }

    /// Render a pixel-perfect circle to our billboard texture.
    fn create_billboard(&self, encoder: &mut wgpu::CommandEncoder) {
        // TODO: This should adapt to changing screen sizes and levels of zoom
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.billboard,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        pass.set_pipeline(&self.billboard_pipeline);
        pass.set_vertex_buffer(0, self.billboard_verts.slice(..));
        pass.draw(0..6, 0..1);
    }
}

impl CircleSender {
    pub fn push_circle(
        &self,
        pos: Point,
        radius: f32,
    ) -> Result<(), mpsc::SendError<(Point, f32)>> {
        self.circle_tx.send((pos, radius))
    }
}
