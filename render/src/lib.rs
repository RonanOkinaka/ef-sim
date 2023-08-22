//! Re-export the relevant (i.e. public) parts of modules from this workspace.

mod window;
mod line;

pub use window::*;
pub use line::*;

use pollster::block_on;

async fn run_async() -> ! {
    let window = Window::with_size("This Should Work...", 640, 480);

    let instance = wgpu::Instance::default();

    // Unfortunate but unavoidable
    let surface = unsafe { instance.create_surface(&window) }.unwrap();
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

    let mut renderer = LineRenderer::new(&device, &adapter, &surface);

    window.run(
        move || {
            let curr_frame = surface
                .get_current_texture()
                .expect("Failed to retrieve swapchain texture");
            let frame_view = curr_frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            renderer.render(&device, &queue, &frame_view);

            curr_frame.present();
        },
        |_| { }
    );
}

/// Temporary helper function
pub fn run() -> ! {
    block_on(run_async());
}
