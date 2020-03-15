use std::sync::mpsc::{channel, Sender};
use std::thread;

use rug::Float;
use threadpool::ThreadPool;

use glium::{
    glutin::{
        self,
        event::{ElementState, Event, ModifiersState, MouseButton, MouseScrollDelta, WindowEvent},
        event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    },
    Surface,
};
use imgui::{Condition, Context, FontConfig, FontGlyphRanges, FontSource};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use num_traits::{FromPrimitive, ToPrimitive};

use crate::mandelbrot::{
    bounded::BoundsSettings,
    compute::{Compute, ComputeEngine, ComputeSettings, ComputedSet},
};

use crate::ui::{events::ComputeEvent, render::AppRenderer};

#[derive(Clone)]
pub struct AppSettings {
    precision: u32,
    resolution: [u32; 2],
    iterations: u64,
    engine: ComputeEngine,
}

impl AppSettings {
    pub fn new() -> AppSettings {
        AppSettings {
            precision: 53,
            resolution: [1600, 900],
            iterations: 1000,
            engine: ComputeEngine::SimdF64x4,
        }
    }
}

#[derive(Clone)]
pub struct ZoomState {
    pos: [Float; 2],
    scale: Float,
}

impl ZoomState {
    fn new(settings: &AppSettings) -> ZoomState {
        ZoomState {
            pos: [
                Float::with_val(settings.precision, -0.5),
                Float::with_val(settings.precision, 0.0),
            ],
            scale: Float::with_val(settings.precision, 1.75),
        }
    }

    fn get_x(&self) -> &Float {
        &self.pos[0]
    }

    fn get_y(&self) -> &Float {
        &self.pos[1]
    }

    fn get_scale(&self) -> &Float {
        &self.scale
    }

    fn set_by_dragging(&mut self, start: [f64; 2], end: [f64; 2], settings: &AppSettings) {
        let scale_xy = [(start[0] - end[0]).abs(), (start[1] - end[1]).abs()];
        let ratio = Float::with_val(settings.precision, settings.resolution[0])
            / f64::from(settings.resolution[1]);
        let scale = Float::with_val(settings.precision, scale_xy[1]) * &self.scale;
        let pos = [
            &self.pos[0]
                + Float::with_val(
                    settings.precision,
                    ((start[0] + end[0]) - 1.0) / 2.0 * self.scale.clone() * ratio,
                ),
            &self.pos[1]
                - Float::with_val(
                    settings.precision,
                    ((start[1] + end[1]) - 1.0) / 2.0 * self.scale.clone(),
                ),
        ];
        self.pos = pos;
        self.scale = scale;
    }

    fn zoom_position(&mut self, pos: [f64; 2], scale: f64, settings: &AppSettings) {
        self.scale *= scale;
        let ratio = Float::with_val(settings.precision, settings.resolution[0])
            / f64::from(settings.resolution[1]);
        let pos = [
            &self.pos[0]
                + Float::with_val(
                    settings.precision,
                    (pos[0] - 0.5) * self.scale.clone() * ratio,
                ),
            &self.pos[1] - Float::with_val(settings.precision, (pos[1] - 0.5) * self.scale.clone()),
        ];
        self.pos = pos;
    }

    fn zoom_scale(&mut self, scale: f64) {
        self.scale *= scale
    }
}

pub struct AppState {
    pub computed_set: ComputedSet,
    pub set_valid: bool,
    pub progress: ComputeEvent,

    pub mouse_pos: [f64; 2],
    pub dragging: bool,
    pub mouse_start: [f64; 2],
    pub mouse_end: [f64; 2],
    pub modifiers: ModifiersState,
    pub zoomstate: ZoomState,
    pub compute_valid: bool,
    pub compute_busy: bool,

    pub compute_start: Option<std::time::Instant>,
    pub compute_time: Option<std::time::Duration>,
}

impl AppState {
    fn new(settings: &AppSettings) -> AppState {
        AppState {
            computed_set: ComputedSet::empty(64, 64),
            set_valid: false,
            progress: ComputeEvent::End,

            mouse_pos: [0.0, 0.0],
            dragging: false,
            mouse_start: [0.0, 0.0],
            mouse_end: [0.0, 0.0],
            modifiers: ModifiersState::empty(),
            zoomstate: ZoomState::new(settings),
            compute_valid: false,
            compute_busy: false,

            compute_start: None,
            compute_time: None,
        }
    }
}

pub struct App {
    event_loop: EventLoop<()>,
    display: glium::Display,
    imgui: Context,
    imgui_platform: WinitPlatform,
    app_render: AppRenderer,

    state: AppState,
    settings: AppSettings,
}

impl App {
    pub fn new(settings: AppSettings) -> App {
        let event_loop = EventLoop::new();
        let context = glutin::ContextBuilder::new().with_vsync(true);
        let builder = glutin::window::WindowBuilder::new()
            .with_title("mandelbrot explorer")
            .with_inner_size(glutin::dpi::LogicalSize::new(1600f64, 900f64));
        let display = glium::Display::new(builder, context, &event_loop).unwrap();

        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        let hidpi_factor = display.gl_window().window().scale_factor();
        let font_size = 13.0 * hidpi_factor as f32;

        imgui.fonts().add_font(&[
            FontSource::DefaultFontData {
                config: Some(FontConfig {
                    size_pixels: font_size,
                    ..FontConfig::default()
                }),
            },
            FontSource::TtfData {
                data: include_bytes!("../../mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 1.75,
                    glyph_ranges: FontGlyphRanges::japanese(),
                    ..FontConfig::default()
                }),
            },
        ]);

        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        imgui.io_mut().display_size = [1600f32, 900f32];

        let mut platform = WinitPlatform::init(&mut imgui);
        {
            let gl_window = display.gl_window();
            let window = gl_window.window();
            platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);
        }

        let app_render = AppRenderer::init();

        let state = AppState::new(&settings);

        App {
            event_loop: event_loop,
            display: display,
            imgui: imgui,
            imgui_platform: platform,
            app_render: app_render,
            state: state,
            settings: settings,
        }
    }

    fn recompute(
        zoomstate: &ZoomState,
        settings: &AppSettings,
        tx: Sender<ComputedSet>,
        update_tx: Sender<ComputeEvent>,
    ) -> thread::JoinHandle<()> {
        let prec = settings.precision;
        let x = Float::with_val(prec, zoomstate.get_x());
        let y = Float::with_val(prec, zoomstate.get_y());
        let scale = Float::with_val(prec, zoomstate.get_scale());
        let [w, h] = settings.resolution;
        let engine = settings.engine;
        let iterations = settings.iterations;
        thread::spawn(move || {
            tx.send(Compute::compute_set(
                Some(&mut ThreadPool::new(8)),
                Some(update_tx),
                &ComputeSettings::new(
                    x,
                    y,
                    scale,
                    w,
                    h,
                    engine,
                    BoundsSettings::new(iterations, prec),
                ),
            ))
            .unwrap();
        })
    }

    pub fn run(self) {
        let (tx, rx) = channel();
        let (compute_tx, compute_rx) = channel();

        self.display.gl_window().window().set_maximized(false);

        let mut frame_time = std::time::Instant::now();

        let App {
            display,
            mut state,
            mut imgui,
            mut settings,
            event_loop,
            mut app_render,
            mut imgui_platform,
            ..
        } = self;

        let mut renderer = imgui_glium_renderer::Renderer::init(&mut imgui, &display).unwrap();

        event_loop.run(
            move |event: Event<()>, _target: &EventLoopWindowTarget<()>, flow: &mut ControlFlow| {
                let gl_window = display.gl_window();
                imgui_platform.handle_event(imgui.io_mut(), gl_window.window(), &event);

                match event {
                    Event::NewEvents(_) => {
                        imgui.io_mut().update_delta_time(frame_time);
                    }
                    Event::MainEventsCleared => {
                        let gl_window = display.gl_window();
                        imgui_platform
                            .prepare_frame(imgui.io_mut(), &gl_window.window())
                            .unwrap();
                        gl_window.window().request_redraw();
                    }
                    Event::RedrawRequested(_) => {
                        Self::redraw(
                            &display,
                            &mut imgui,
                            &mut frame_time,
                            &mut app_render,
                            &mut state,
                            &mut settings,
                            &mut renderer,
                        );
                    }
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        *flow = ControlFlow::Exit;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::CursorMoved { position, .. },
                        ..
                    } => {
                        if !imgui.io().want_capture_mouse {
                            let size = display.gl_window().window().inner_size();
                            state.mouse_pos = [
                                position.x / size.width as f64,
                                position.y / size.height as f64,
                            ];
                            state.mouse_end = state.mouse_pos;
                        }
                    }
                    Event::WindowEvent {
                        event:
                            WindowEvent::MouseInput {
                                state: mouse_state,
                                button: MouseButton::Left,
                                ..
                            },
                        ..
                    } => {
                        if !imgui.io().want_capture_mouse {
                            match mouse_state {
                                ElementState::Pressed => {
                                    if !state.compute_busy {
                                        state.mouse_start = state.mouse_pos;
                                        state.dragging = true;
                                    }
                                }
                                ElementState::Released => {
                                    if !state.compute_busy && state.dragging {
                                        state.mouse_end = state.mouse_pos;
                                        state.dragging = false;
                                        let start = state.mouse_start;
                                        let end = state.mouse_end;
                                        if (end[1] - start[1]) + (end[0] - start[0]) > 0.001 {
                                            state.zoomstate.set_by_dragging(start, end, &settings);
                                        } else {
                                            state.zoomstate.zoom_position(start, {
                                                if state.modifiers.shift() { 0.7 } else { 1.0 }
                                            }, &settings);
                                        }
                                        state.compute_valid = false;
                                    }
                                }
                            }
                        }
                    }
                    Event::WindowEvent {
                        event:
                            WindowEvent::MouseWheel {
                                delta: MouseScrollDelta::LineDelta(_delta_x, delta_y),
                                ..
                            },
                        ..
                    } => {
                        if !state.compute_busy {
                            let m = if state.modifiers.shift() { 3.0 } else { 1.5 };
                            let scale = 1.0 + (m * -delta_y / 10.0) as f64;
                            if state.modifiers.ctrl() {
                                state.zoomstate.zoom_scale(scale);
                            } else {
                                state.zoomstate.zoom_position(state.mouse_pos, scale, &settings);
                            }
                            state.compute_valid = false;
                        }
                    }
                    Event::WindowEvent {
                        event: WindowEvent::ModifiersChanged(modifiers),
                        ..
                    } => {
                        state.modifiers = modifiers;
                    }
                    _ => {}
                }

                if !state.compute_valid {
                    App::recompute(&state.zoomstate, &settings, tx.clone(), compute_tx.clone());
                    state.compute_valid = true;
                    state.compute_busy = true;
                    state.compute_start = Some(std::time::Instant::now());
                    state.compute_time = None;
                }

                if let Ok(result) = rx.try_recv() {
                    state.computed_set = result;
                    state.set_valid = false;
                    state.compute_busy = false;
                    state.compute_time = Some(state.compute_start.unwrap().elapsed());
                    state.compute_start = None;
                }

                for event in compute_rx.try_iter() {
                    state.progress = event;
                }
            },
        );
    }

    fn redraw(
        display: &glium::Display,
        imgui: &mut imgui::Context,
        frame_time: &mut std::time::Instant,
        app_render: &mut AppRenderer,
        state: &mut AppState,
        settings: &mut AppSettings,
        renderer: &mut imgui_glium_renderer::Renderer,
    ) {
        let io = imgui.io_mut();
        //platform.borrow().prepare_frame(io, &window).unwrap();
        *frame_time = io.update_delta_time(*frame_time);

        let mut target = display.draw();
        target.clear_color_srgb(1.0, 1.0, 1.0, 1.0);
        app_render.render(state, &mut target, display);
        //platform.borrow().prepare_render(&ui, &window);
        let ui = imgui.frame();
        Self::build_ui(&ui, state, settings);
        //render ui
        let draw_data = ui.render();
        // render mandelbrot

        //render imgui ui to glium
        renderer.render(&mut target, draw_data).unwrap();
        //swap buffers
        target.finish().unwrap();
    }

    fn build_ui(ui: &imgui::Ui, state: &mut AppState, settings: &mut AppSettings) {
        imgui::Window::new(im_str!("Mandelbrot-explorer"))
            .size([400.0, 600.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!(
                    "Position:\n\tX:{:.4}\n\tY:{:.4}",
                    state.zoomstate.get_x(),
                    state.zoomstate.get_y()
                ));
                ui.separator();
                ui.text(im_str!("Scale:\n\t{:.4}", state.zoomstate.get_scale()));
                ui.separator();
                if ui.button(im_str!("Render"), [60.0, 20.0]) && !state.compute_busy {
                    state.compute_valid = false;
                };
                if ui.button(im_str!("Reset"), [60.0, 20.0]) && !state.compute_busy {
                    state.zoomstate = ZoomState::new(&settings);
                    state.compute_valid = false;
                }
                ui.separator();
                let mut iterations = settings.iterations as i32;
                ui.input_int(im_str!("Iterations"), &mut iterations).build();
                settings.iterations = iterations as u64;
                ui.separator();
                let items: Vec<_> = ComputeEngine::LIST
                    .iter()
                    .map(|x| im_str!("{:?}", x))
                    .collect();
                let mut select: i32 = settings.engine.to_i32().unwrap();
                if ui.list_box(
                    im_str!("Engine"),
                    &mut select,
                    items.iter().collect::<Vec<_>>().as_slice(),
                    items.len() as i32,
                ) {
                    settings.engine = FromPrimitive::from_i32(select).unwrap()
                }
                ui.separator();
                let mut precision = settings.precision as i32;
                ui.input_int(im_str!("Precision bits"), &mut precision)
                    .build();
                settings.precision = precision as u32;
                ui.separator();
                imgui::ProgressBar::new(match state.progress {
                    ComputeEvent::Progress((a, b)) => a as f32 / b as f32,
                    _ => 0f32,
                })
                .build(&ui);

                ui.separator();
                ui.text(im_str!("Render time:"));
                if let Some(duration) = state.compute_time {
                    ui.text(im_str!("\t{:.4} seconds", duration.as_secs_f64()));
                } else {
                    ui.text(im_str!("\tn/a"));
                }
                ui.separator();
                ui.text(im_str!(r"
Area drag: zoom in on area
click: move to position
shift+click: click zoom in on position
ctrl+scroll: zoom in on center
scroll: zoom in and move to position
hold shift: zoom more
                "))
            });
    }
}
