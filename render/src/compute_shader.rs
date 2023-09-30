//! Abstraction for dealing with compute shaders.

pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroup_size_x: u32,
}

pub enum ComputePipelineBuffer<'a> {
    Uniform(&'a wgpu::Buffer),
    Storage(&'a wgpu::Buffer),
    StorageReadOnly(&'a wgpu::Buffer),
}

#[allow(dead_code)]
impl ComputePipeline {
    pub fn new(
        device: &wgpu::Device,
        buffers: &[ComputePipelineBuffer],
        shader: &wgpu::ShaderModule,
        entry_point: &str,
        workgroup_size_x: u32,
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
                ComputePipelineBuffer::StorageReadOnly(inner) => {
                    layout_entries.push(wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
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
            module: shader,
            entry_point,
        });

        Self {
            pipeline,
            bind_group,
            workgroup_size_x,
        }
    }

    pub fn run(&self, encoder: &mut wgpu::CommandEncoder, invocations: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        self.bind_self(&mut pass);
        self.run_shared(&mut pass, invocations);
    }

    pub fn run_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &wgpu::Buffer,
        offset: u64,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        self.bind_self(&mut pass);
        self.run_indirect_shared(&mut pass, params, offset);
    }

    pub fn run_shared<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, invocations: u32) {
        // Round up the number of workgroups
        let num_work_groups = (invocations + self.workgroup_size_x - 1) / self.workgroup_size_x;

        self.bind_self(pass);
        pass.dispatch_workgroups(num_work_groups, 1, 1);
    }

    pub fn run_indirect_shared<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        params: &'a wgpu::Buffer,
        offset: u64,
    ) {
        self.bind_self(pass);
        pass.dispatch_workgroups_indirect(params, offset);
    }

    fn bind_self<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
    }
}
