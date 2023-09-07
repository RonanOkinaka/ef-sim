//! Define the type responsible for rendering lines.

use std::sync::mpsc;
use util::math::Point;

/// State required for rendering lines.
pub struct LineRenderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,

    compute_command_buf: wgpu::Buffer,
    compute_binding: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,

    rx: mpsc::Receiver<Point>,
    num_indices: u64,
}

/// State required for requesting a line be rendered.
pub struct LineSender {
    tx: mpsc::Sender<Point>,
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

        let points: Vec<Point> = self.rx.try_iter().collect();

        if !points.is_empty() {
            queue.write_buffer(
                &self.compute_command_buf,
                0,
                bytemuck::cast_slice(points.as_slice()),
            );

            // TODO: temporary hack, this will be replaced with compute-driven
            // indirect draws later
            if self.num_indices == 1 {
                self.num_indices = 0;
            } else {
                self.num_indices += 6 * points.len() as u64;
            }

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_binding, &[]);
            pass.dispatch_workgroups(points.len() as u32, 1, 1);
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
        rx: mpsc::Receiver<Point>,
    ) -> Self {
        // Create the index and vertex buffers
        let vertex_buf = create_buffer_with_intro(device, wgpu::BufferUsages::VERTEX, &[0]);
        let index_buf = create_buffer_with_intro(device, wgpu::BufferUsages::INDEX, &[0]);

        // Load the shaders
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "draw_line.wgsl"
            ))),
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
        let bind_group_layout_entry_default = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        ..bind_group_layout_entry_default
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        ..bind_group_layout_entry_default
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        ..bind_group_layout_entry_default
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        ..bind_group_layout_entry_default
                    },
                ],
            });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "compute_curve.wgsl"
            ))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_layout),
            module: &compute_shader,
            entry_point: "compute_main",
        });

        // Create the compute shader buffers
        let compute_command_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16384,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let curve_buf = create_buffer_with_intro(device, wgpu::BufferUsages::COPY_DST, &[-1; 10]);

        let compute_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &compute_command_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &vertex_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &index_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &curve_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Self {
            vertex_buf,
            index_buf,
            render_pipeline,
            compute_command_buf,
            compute_binding,
            compute_pipeline,
            rx,
            num_indices: 1,
        }
    }
}

impl LineSender {
    pub fn push_point(&self, point: Point) -> Result<(), mpsc::SendError<Point>> {
        self.tx.send(point)
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
