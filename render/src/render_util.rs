//! Various utilities for rendering.

use crate::window::Window;

pub trait Renderer {
    /// Given a device, queue, and a texture to draw to, return a CommandBuffer
    /// that renders the class' data.
    /// CommandBuffer comes from encoder.finish().
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
    ) -> wgpu::CommandBuffer;

    // TODO: Resize handling goes here
}

pub struct RenderGraph {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface,

    // TODO: This will become a DAG, but this is fine for now
    renderers: Vec<Box<dyn Renderer>>,
}

impl RenderGraph {
    pub async fn from_window(window: &Window) -> Self {
        let instance = wgpu::Instance::default();

        // Unfortunate but unavoidable
        let surface = unsafe { instance.create_surface(window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let (width, height) = window.get_size();
        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        Self {
            adapter,
            device,
            queue,
            surface,
            renderers: Vec::new(),
        }
    }

    pub fn add_renderer<T: Renderer + 'static>(&mut self, renderer: T) {
        self.renderers.push(Box::new(renderer));
    }

    pub fn render(&mut self) {
        let curr_frame = self
            .surface
            .get_current_texture()
            .expect("Failed to retrieve swapchain texture");
        let frame_view = curr_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let commands = self
            .renderers
            .iter_mut()
            .map(|renderer| renderer.render(&self.device, &self.queue, &frame_view));
        self.queue.submit(commands);

        curr_frame.present();
    }
}
