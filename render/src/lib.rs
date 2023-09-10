//! Re-export the relevant (i.e. public) parts of modules from this workspace.

mod line;
mod window;

pub use line::*;
pub use window::*;

use pollster::block_on;
use util::math::Point;

async fn run_async() -> ! {
    let window = Window::with_size("This Should Work...", 640, 480);

    // TODO: Hard-coded for now
    let (mx, my): (f32, f32) = (2. / 640., 2. / 480.);

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

    let (sender, mut renderer) = line_renderer(&device, &adapter, &surface);
    let mut curve_index = 0;

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
        move |input_event| {
            match input_event.reason {
                InputEventType::MouseLeft => {
                    let cursor = input_event.cursor;
                    if cursor.left_down {
                        let pos = Point(cursor.x * mx - 1.0, 1.0 - cursor.y * my);
                        sender
                            .push_point(pos, curve_index)
                            .expect("Render thread should never hang up");
                    }
                }
                InputEventType::KeyboardButton(true, key) => {
                    match key {
                        // TODO: This could probably be a map and a macro...
                        VirtualKeyCode::Key1 => curve_index = 0,
                        VirtualKeyCode::Key2 => curve_index = 1,
                        VirtualKeyCode::Key3 => curve_index = 2,
                        VirtualKeyCode::Key4 => curve_index = 3,
                        VirtualKeyCode::Key5 => curve_index = 4,
                        _ => {}
                    }
                }
                _ => {}
            }
        },
    );
}

pub fn run() -> ! {
    block_on(run_async());
}
