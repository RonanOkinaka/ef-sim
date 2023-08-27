//! Define the type responsible for rendering lines.

use bytemuck::cast_slice;
use std::mem::size_of;
use std::sync::mpsc;
use util::math::{almost_zero, Point};

/// State required for rendering lines.
pub struct LineRenderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,

    render_pipeline: wgpu::RenderPipeline,

    rx: mpsc::Receiver<Point>,
    num_verts: u64,
    num_indices: u64,
    last_pos: Option<Point>,
    last_dir: Option<Point>,
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
    (
        LineSender { tx },
        LineRenderer::with_channel(device, adapter, surface, rx),
    )
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

        self.update_buffers(queue);

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
            pass.draw_indexed(0..self.num_indices as u32, 0, 0..1);
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
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 10000 * size_of::<Point>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 10000 * size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        // Load the shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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

        Self {
            vertex_buf,
            index_buf,
            render_pipeline,
            rx,
            num_verts: 0,
            num_indices: 0,
            last_dir: None,
            last_pos: None,
        }
    }

    fn extend_segment(from: Point, to: Point) -> Point {
        (to - from).orthogonalize().normalize() * 0.01
    }

    fn merge_segment_dirs(dir1: Point, dir2: Point) -> Point {
        let u = dir1.normalize();
        let v = dir2.normalize();

        let det = u.reverse_orthogonalize().dot(v);

        if almost_zero(det) {
            // TODO: Don't just give up
            panic!("Angle join will extend to infinity");
        }

        let mul = v.dot(v - u) / det;
        (u.reverse_orthogonalize() * mul + u) * 0.01
    }

    fn update_buffers(&mut self, queue: &wgpu::Queue) {
        let mut rx_iter = self.rx.try_iter();

        if self.last_pos.is_none() {
            // If this is the first point, just update our state
            self.last_pos = rx_iter.next();
            if self.last_pos.is_none() {
                return;
            }
        }

        let mut new_verts: Vec<Point> = Vec::new();
        let mut new_indices: Vec<u32> = Vec::new();
        let prev_num_verts = self.num_verts;
        let prev_num_indices = self.num_indices;

        if self.last_dir.is_none() {
            let new_pos = match rx_iter.next() {
                Some(new_pos) => new_pos,
                None => return,
            };

            let last_pos = self.last_pos.unwrap();

            // If this is the second point, don't merge their directions
            let new_dir = Self::extend_segment(last_pos, new_pos);

            // Push the previous
            new_verts.push(last_pos + new_dir);
            new_verts.push(last_pos - new_dir);
            new_indices.extend_from_slice(&[0, 1, 2, 3, 2, 1]);

            self.num_verts = 4;
            self.num_indices = 6;
            self.last_pos = Some(new_pos);
            self.last_dir = Some(new_dir);
        }

        // For all other points, proceed in the same way
        let mut last_pos = self.last_pos.unwrap();
        let mut last_dir = self.last_dir.unwrap();
        for new_pos in rx_iter {
            // Calculate the new direction
            let new_dir = Self::extend_segment(last_pos, new_pos);

            // Merge it with the last
            let mid_dir = Self::merge_segment_dirs(new_dir, last_dir);

            // Push the previous
            new_verts.push(last_pos + mid_dir);
            new_verts.push(last_pos - mid_dir);

            let index = self.num_verts as u32;
            new_indices.extend_from_slice(&[
                index - 2, // 0, 1, 2
                index - 1,
                index,
                index + 1, // 3, 2, 1
                index,
                index - 1,
            ]);

            // Update for next iteration
            self.num_verts += 2;
            self.num_indices += 6;
            last_pos = new_pos;
            last_dir = new_dir;
        }

        if new_verts.is_empty() {
            return;
        }

        // Finally, push the last
        new_verts.push(last_pos + last_dir);
        new_verts.push(last_pos - last_dir);
        self.last_pos = Some(last_pos);
        self.last_dir = Some(last_dir);

        let vertex_write_start = if prev_num_verts > 2 {
            // Overwrite the previous two vertices, if they exist
            prev_num_verts - 2
        } else {
            prev_num_verts
        };

        queue.write_buffer(
            &self.vertex_buf,
            vertex_write_start * size_of::<Point>() as u64,
            cast_slice(new_verts.as_slice()),
        );
        queue.write_buffer(
            &self.index_buf,
            prev_num_indices * size_of::<u32>() as u64,
            cast_slice(new_indices.as_slice()),
        );
    }
}

impl LineSender {
    pub fn push_point(&self, point: Point) -> Result<(), mpsc::SendError<Point>> {
        self.tx.send(point)
    }
}
