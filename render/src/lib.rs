//! Re-export the relevant (i.e. public) parts of modules from this workspace.

mod circle;
mod compute_shader;
mod line;
mod render_util;
mod shader;
mod window;

pub use line::*;
pub use window::*;

use pollster::block_on;
use util::math::Point;

async fn run_async() -> ! {
    let window = Window::with_size("This Should Work...", 640, 480);

    let mut render = render_util::RenderGraph::from_window(&window).await;

    let (sender, particle_renderer) =
        particle_renderer(&render.device, &render.adapter, &render.surface);
    sender.push_charge(Point(-0.5, 0.0), 1.0).unwrap();
    sender.push_charge(Point(0.5, 0.0), -1.0).unwrap();

    let mut circle_renderer =
        circle::CircleRenderer::new(&render.device, &render.adapter, &render.surface, 100);
    circle_renderer.push_circle(Point(-0.5, 0.0), 0.1, &render.queue);
    circle_renderer.push_circle(Point(0.5, 0.0), 0.1, &render.queue);

    render.add_renderer(particle_renderer);
    render.add_renderer(circle_renderer);

    window.run(move || render.render(), move |_input_event| {});
}

pub fn run() -> ! {
    block_on(run_async());
}
