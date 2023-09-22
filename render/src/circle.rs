//! Renderer for drawing solid-color, billboarded circles.

use bytemuck::cast_slice;
use util::math::Point;
use wgpu::util::DeviceExt;

use crate::render_util::Renderer;
use crate::shader::WgslLoader;

pub struct CircleRenderer {
    num_verts: u32,
    vertices: wgpu::Buffer,

    billboard: wgpu::TextureView,
    billboard_verts: wgpu::Buffer,
    billboard_pipeline: wgpu::RenderPipeline,
    billboard_ready: bool,

    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
}

const VERTEX_SIZE: u64 = 4 * 4;

impl Renderer for CircleRenderer {
    fn render(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if !self.billboard_ready {
            self.billboard_ready = true;
            self.create_billboard(&mut encoder);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
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
            pass.draw(0..self.num_verts, 0..1);
        }

        encoder.finish()
    }
}

impl CircleRenderer {
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        surface: &wgpu::Surface,
        max_num_billboards: u32,
    ) -> Self {
        // Create the billboard generation pipeline
        let best_format = surface.get_capabilities(adapter).formats[0];

        let billboard_texture = device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
        let shader = shader_loader.create_shader(device, include_str!("draw_circle_texture.wgsl"));

        let billboard_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let billboard_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let billboard_verts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: VERTEX_SIZE * 6 * max_num_billboards as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader =
            shader_loader.create_shader(device, include_str!("draw_circle_billboards.wgsl"));

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    ..surface.get_capabilities(adapter).formats[0].into()
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            num_verts: 0,
            billboard,
            billboard_verts,
            billboard_pipeline,
            billboard_ready: false,
            render_pipeline,
            render_bind_group,
        }
    }

    pub fn push_circle(&mut self, pos: Point, width: f32, queue: &wgpu::Queue) {
        let right = pos.0 + width;
        let left = pos.0 - width;
        let top = pos.1 + width;
        let bottom = pos.1 - width;

        #[rustfmt::skip]
        queue.write_buffer(
            &self.vertices,
            VERTEX_SIZE * self.num_verts as u64,
            cast_slice(&[
                right, top, 1.0, 0.0,
                left, top, 0.0, 0.0,
                left, bottom, 0.0, 1.0,
                left, bottom, 0.0, 1.0,
                right, bottom, 1.0, 1.0,
                right, top, 1.0, 0.0,
            ]),
        );
        self.num_verts += 6;
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
