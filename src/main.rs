use render::{InputEvent, Window};

fn main() {
    let window = Window::with_size("Interesting Title", 640, 480);
    window.run(
        || {},
        |event: InputEvent| {
            println!("{:?}", event);
        },
    );
}
