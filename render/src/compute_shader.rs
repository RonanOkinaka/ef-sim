//! Abstraction for dealing with compute shaders.

pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

pub enum ComputePipelineBuffer<'a> {
    Uniform(&'a wgpu::Buffer),
    Storage(&'a wgpu::Buffer),
}

impl ComputePipeline {
    pub fn new(
        device: &wgpu::Device,
        buffers: &[ComputePipelineBuffer],
        shader: wgpu::ShaderModule,
        entry_point: &str,
    ) -> Self {
        let uniform_binding = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage_binding = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };

        let mut layout_entries = Vec::new();
        let mut binding_entries = Vec::new();

        // Generate the bindings automatically
        // (This is possible to simplify because I use few features)
        for (binding_index, buffer) in buffers.iter().enumerate() {
            let binding = binding_index as u32;

            let buffer = match buffer {
                ComputePipelineBuffer::Uniform(inner) => {
                    layout_entries.push(wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: uniform_binding,
                        count: None,
                    });
                    inner
                }
                ComputePipelineBuffer::Storage(inner) => {
                    layout_entries.push(wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: storage_binding,
                        count: None,
                    });
                    inner
                }
            };

            binding_entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: layout_entries.as_slice(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: binding_entries.as_slice(),
        });

        // Create the pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
        });

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, work_groups: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(work_groups, 1, 1);
    }
}
