use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use std::default::Default;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::{Window as WinitWindow, WindowBuilder},
};

/// Simplified cursor model containing its position and three major button states.
#[derive(Clone, Copy, Debug, Default)]
pub struct MouseData {
    pub left_down: bool,
    pub right_down: bool,
    pub middle_down: bool,
    pub x: f32,
    pub y: f32,
}

pub struct Window {
    ev_loop: EventLoop<()>,
    window: WinitWindow,
    cursor: MouseData,
}

/// We simply re-export Winit's virtual keyboard enum.
pub use winit::event::VirtualKeyCode;

#[derive(Clone, Copy, Debug)]
pub enum InputEventType {
    MouseMoved,
    MouseLeft,
    MouseMiddle,
    MouseRight,
    KeyboardButton(bool, VirtualKeyCode),
}

#[derive(Clone, Copy, Debug)]
pub struct InputEvent {
    pub reason: InputEventType,
    pub cursor: MouseData,
}

impl Window {
    pub fn run<F1, F2>(mut self, mut on_redraw: F1, mut on_input: F2) -> !
    where
        F1: FnMut() + 'static,
        F2: FnMut(InputEvent) + 'static,
    {
        self.ev_loop.run(move |event, _, control_flow| {
            match event {
                Event::RedrawRequested(window_id) if window_id == self.window.id() => on_redraw(),
                Event::WindowEvent { window_id, event } if window_id == self.window.id() => {
                    match event {
                        WindowEvent::CloseRequested => control_flow.set_exit(),
                        // Everything below this is just input translation
                        WindowEvent::MouseInput { state, button, .. } => {
                            let pressed = state == ElementState::Pressed;
                            let mouse_button;

                            match button {
                                MouseButton::Left => {
                                    self.cursor.left_down = pressed;
                                    mouse_button = InputEventType::MouseLeft;
                                }
                                MouseButton::Middle => {
                                    self.cursor.middle_down = pressed;
                                    mouse_button = InputEventType::MouseMiddle;
                                }
                                MouseButton::Right => {
                                    self.cursor.right_down = pressed;
                                    mouse_button = InputEventType::MouseRight;
                                }
                                _ => return,
                            }

                            on_input(InputEvent {
                                reason: mouse_button,
                                cursor: self.cursor,
                            });
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            self.cursor.x = position.x as f32;
                            self.cursor.y = position.y as f32;

                            on_input(InputEvent {
                                reason: InputEventType::MouseMoved,
                                cursor: self.cursor,
                            });
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if let Some(key) = input.virtual_keycode {
                                let pressed = input.state == ElementState::Pressed;
                                on_input(InputEvent {
                                    reason: InputEventType::KeyboardButton(pressed, key),
                                    cursor: self.cursor,
                                });
                            }
                        }
                        _ => {} // TODO: Must implement window resizing for the renderer
                    }
                }
                Event::RedrawEventsCleared => self.window.request_redraw(),
                _ => {}
            }
        });
    }

    // Note: I hope to support WASM later, hence the config attribute
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_size<T>(title: T, width: u32, height: u32) -> Self
    where
        T: Into<String>,
    {
        let ev_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(PhysicalSize { width, height })
            .build(&ev_loop)
            .expect("Window creation failed");

        Self {
            ev_loop,
            window,
            cursor: Default::default(),
        }
    }

    pub fn get_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }
}

/// Facilitates interop with WGPU
unsafe impl HasRawDisplayHandle for Window {
    fn raw_display_handle(&self) -> RawDisplayHandle {
        self.window.raw_display_handle()
    }
}

/// Facilitates interop with WGPU
unsafe impl HasRawWindowHandle for Window {
    fn raw_window_handle(&self) -> RawWindowHandle {
        self.window.raw_window_handle()
    }
}
