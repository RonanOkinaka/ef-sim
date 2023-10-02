//! Re-export the relevant (i.e. public) parts of modules from this workspace.

mod circle;
mod compute_shader;
mod line;
mod render_util;
mod shader;
mod update_queue;
mod window;

pub use circle::*;
pub use line::*;
pub use window::*;

use pollster::block_on;
use util::math::Point;

fn push_charge(
    pos: Point,
    charge: f32,
    transform: Point,
    circle: &CircleSender,
    particle: &ParticleSender,
) -> u32 {
    let pos = Point(pos.0 * transform.0 - 1.0, 1.0 - pos.1 * transform.1);
    circle.push_circle(pos, 0.1).unwrap();
    match particle.push_charge(pos, charge) {
        Ok(index) => index,
        Err(..) => panic!("Out of memory"),
    }
}

async fn run_async() -> ! {
    let window = Window::with_size("This Should Work...", 640, 480);
    let transform = Point(2. / 640., 2. / 480.);

    let mut render = render_util::RenderGraph::from_window(&window).await;

    let (particle_sender, particle_renderer) =
        particle_renderer(&render.device, &render.adapter, &render.surface);

    // TODO: Choose a maximum value intelligently
    let (circle_sender, circle_renderer) =
        circle::circle_renderer(&render.device, &render.adapter, &render.surface, 100);

    render.add_renderer(particle_renderer);
    render.add_renderer(circle_renderer);

    // Help us test changing particle quantities
    println!("Enter the target quantity of particles and press enter.");
    let thread_particle_sender = particle_sender.clone();
    std::thread::spawn(move || loop {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();

        match line.trim_end().parse::<u32>() {
            Ok(num_curves) => thread_particle_sender.set_num_curves(num_curves),
            Err(..) => println!("Please enter a positive integer!"),
        }
    });

    let mut charge_stack = std::collections::VecDeque::new();
    window.run(
        move || render.render(),
        move |input_event| match input_event {
            InputEvent {
                reason: InputEventType::MouseLeft,
                cursor,
            } if cursor.left_down => {
                charge_stack.push_back(push_charge(
                    Point(cursor.x, cursor.y),
                    1.0,
                    transform,
                    &circle_sender,
                    &particle_sender,
                ));
            }
            InputEvent {
                reason: InputEventType::MouseRight,
                cursor,
            } if cursor.right_down => {
                charge_stack.push_back(push_charge(
                    Point(cursor.x, cursor.y),
                    -1.0,
                    transform,
                    &circle_sender,
                    &particle_sender,
                ));
            }
            InputEvent {
                reason: InputEventType::KeyboardButton(true, VirtualKeyCode::Space),
                ..
            } => {
                if let Some(key) = charge_stack.pop_front() {
                    particle_sender.pop_charge(key).unwrap();
                }
            }
            _ => {}
        },
    );
}

pub fn run() -> ! {
    block_on(run_async());
}
