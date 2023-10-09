//! Re-export the relevant (i.e. public) parts of modules from this workspace.

mod circle;
mod compute_shader;
mod line;
mod particle_simulation;
mod render_util;
mod shader;
mod update_queue;
mod window;

use particle_simulation::*;
pub use window::*;

use pollster::block_on;
use render_util::*;
use util::math::Point;

async fn run_async() -> ! {
    let window = Window::with_size("This Should Work...", 640, 480);
    let transform = Point(2. / 640., 2. / 480.);

    let mut render = RenderGraph::from_window(&window).await;
    let sender = particle_simulation(&mut render);

    sender.set_particle_lifetime(10.0);

    // Help us test changing particle quantities
    println!("Enter the target quantity of particles and press enter.");
    let thread_particle_sender = sender.clone();
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
        move || render.tick(),
        move |input_event| match input_event {
            InputEvent {
                reason: InputEventType::MouseLeft,
                cursor,
            } if cursor.left_down => {
                charge_stack.push_back(
                    sender
                        .push_charge(
                            Point(cursor.x * transform.0 - 1.0, 1.0 - cursor.y * transform.1),
                            1.0,
                            0.1,
                        )
                        .unwrap_or_else(|_| panic!("blah")),
                );
            }
            InputEvent {
                reason: InputEventType::MouseRight,
                cursor,
            } if cursor.right_down => {
                charge_stack.push_back(
                    sender
                        .push_charge(
                            Point(cursor.x * transform.0 - 1.0, 1.0 - cursor.y * transform.1),
                            -1.0,
                            0.1,
                        )
                        .unwrap_or_else(|_| panic!("blah")),
                );
            }
            InputEvent {
                reason: InputEventType::KeyboardButton(true, VirtualKeyCode::Space),
                ..
            } => {
                if let Some(key) = charge_stack.pop_front() {
                    sender.pop_charge(key).unwrap();
                }
            }
            _ => {}
        },
    );
}

pub fn run() -> ! {
    block_on(run_async());
}
