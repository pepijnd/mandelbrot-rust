use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{channel, Sender};
use std::thread;

use rug::Float;
use threadpool::ThreadPool;

use glium::{
    glutin::{self, ElementState, Event, MouseButton, WindowEvent},
    Display, Surface,
};
use imgui::{Condition, Context, FontConfig, FontGlyphRanges, FontSource, Ui};
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

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
            iterations: 500,
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
        let ratio = f64::from(settings.resolution[0]) / f64::from(settings.resolution[1]);
        let scale = Float::with_val(settings.precision, scale_xy[1]) * &self.scale;
        let pos = [
            &self.pos[0] + Float::with_val(settings.precision, (start[0] + end[0]) - 1.0) / 2.0
                    * &self.scale
                    * ratio,
            &self.pos[1] - Float::with_val(settings.precision, (start[1] + end[1]) - 1.0) / 2.0
                    * &self.scale,
        ];
        self.pos = pos;
        self.scale = scale;
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
    pub zoomstate: ZoomState,
    pub compute_valid: bool,
    pub compute_busy: bool,
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
            zoomstate: ZoomState::new(settings),
            compute_valid: false,
            compute_busy: false,
        }
    }
}

pub struct App {
    events_loop: Rc<RefCell<glutin::EventsLoop>>,
    display: Rc<RefCell<glium::Display>>,
    imgui: Rc<RefCell<Context>>,
    platform: Rc<RefCell<WinitPlatform>>,
    imgui_render: Rc<RefCell<Renderer>>,
    app_render: Rc<RefCell<AppRenderer>>,

    state: Rc<RefCell<AppState>>,
    settings: Rc<RefCell<AppSettings>>,
}

impl App {
    pub fn new(settings: AppSettings) -> App {
        let events_loop = glutin::EventsLoop::new();
        let context = glutin::ContextBuilder::new().with_vsync(true);
        let builder = glutin::WindowBuilder::new()
            .with_title("mandelbrot explorer")
            .with_dimensions(glutin::dpi::LogicalSize::new(1600f64, 900f64));
        let display = Display::new(builder, context, &events_loop).unwrap();

        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui);
        {
            let gl_window = display.gl_window();
            let window = gl_window.window();
            platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);
        }

        let hidpi_factor = platform.hidpi_factor();
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

        let imgui_render = Renderer::init(&mut imgui, &display).unwrap();
        let app_render = AppRenderer::init();

        let state = AppState::new(&settings);

        App {
            events_loop: Rc::new(RefCell::new(events_loop)),
            display: Rc::new(RefCell::new(display)),
            imgui: Rc::new(RefCell::new(imgui)),
            platform: Rc::new(RefCell::new(platform)),
            imgui_render: Rc::new(RefCell::new(imgui_render)),
            app_render: Rc::new(RefCell::new(app_render)),
            state: Rc::new(RefCell::new(state)),
            settings: Rc::new(RefCell::new(settings)),
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

    pub fn main_loop<F: FnMut(&mut bool, &mut Ui, &mut AppState, &mut AppSettings)>(
        &mut self,
        mut run_ui: F,
    ) {
        let (tx, rx) = channel();
        let (compute_tx, compute_rx) = channel();

        let display = self.display.borrow();
        let gl_window = display.gl_window();
        let window = gl_window.window();

        window.set_maximized(false);

        let mut frame_time = std::time::Instant::now();
        let mut run = true;

        let events_loop = self.events_loop.clone();
        let platform = self.platform.clone();
        let imgui = self.imgui.clone();
        let state = self.state.clone();
        let settings = self.settings.clone();
        let imgui_render = self.imgui_render.clone();
        let app_render = self.app_render.clone();

        while run {
            let mut imgui = imgui.borrow_mut();
            let mut state = state.borrow_mut();
            let mut settings = settings.borrow_mut();
            events_loop.borrow_mut().poll_events(|event| {
                platform
                    .borrow_mut()
                    .handle_event(imgui.io_mut(), &window, &event);
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        run = false;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::CursorMoved { position, .. },
                        ..
                    } => {
                        if !imgui.io().want_capture_mouse {
                            let size = window.get_inner_size().unwrap();
                            state.mouse_pos = [position.x / size.width, position.y / size.height];
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
                                        state.zoomstate.set_by_dragging(start, end, &settings);
                                        state.compute_valid = false;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            });
            if !state.compute_valid {
                App::recompute(&state.zoomstate, &settings, tx.clone(), compute_tx.clone());
                state.compute_valid = true;
                state.compute_busy = true;
            }

            if let Ok(result) = rx.try_recv() {
                state.computed_set = result;
                state.set_valid = false;
                state.compute_busy = false;
            }

            for event in compute_rx.try_iter() {
                state.progress = event;
            }

            let io = imgui.io_mut();
            platform.borrow().prepare_frame(io, &window).unwrap();
            frame_time = io.update_delta_time(frame_time);
            let mut ui = imgui.frame();
            run_ui(&mut run, &mut ui, &mut state, &mut settings);
            let mut target = display.draw();
            target.clear_color_srgb(1.0, 1.0, 1.0, 1.0);
            app_render
                .borrow_mut()
                .render(&mut state, &mut target, &(*display));
            platform.borrow().prepare_render(&ui, &window);
            //render ui
            let draw_data = ui.render();
            // render mandelbrot

            //render imgui ui to glium
            imgui_render
                .borrow_mut()
                .render(&mut target, draw_data)
                .unwrap();
            //swap buffers
            target.finish().unwrap();
        }
    }

    pub fn run(&mut self) {
        self.main_loop(|_run, ui, state, settings| {
            ui.window(im_str!("Mandelbrot-explorer"))
                .size([400.0, 600.0], Condition::FirstUseEver)
                .build(|| {
                    ui.text(im_str!(
                        "Position:\n\tX:{:1}\n\tY:{:1})",
                        state.zoomstate.get_x(),
                        state.zoomstate.get_y()
                    ));
                    ui.separator();
                    ui.text(im_str!("Scale:{:1})", state.zoomstate.get_scale()));
                    ui.separator();
                    if ui.button(im_str!("Render"), [60.0, 20.0]) && !state.compute_busy {
                        state.compute_valid = false;
                    };
                    ui.separator();
                    let mut iterations = settings.iterations as i32;
                    ui.input_int(im_str!("Iterations"), &mut iterations).build();
                    settings.iterations = iterations as u64;
                    ui.separator();
                    let items = [
                        im_str!("Single"),
                        im_str!("Double"),
                        im_str!("Simd f64X4"),
                        im_str!("MPC"),
                    ];
                    let mut select = settings.engine.to_int();
                    if ui.list_box(im_str!("Engine"), &mut select, &items, items.len() as i32) {
                        settings.engine = ComputeEngine::from_int(select);
                    }
                    ui.separator();
                    let mut precision = settings.precision as i32;
                    ui.input_int(im_str!("MPC Precision"), &mut precision)
                        .build();
                    settings.precision = precision as u32;
                    ui.separator();
                    match state.progress {
                        ComputeEvent::Progress((a, b)) => {
                            ui.progress_bar(a as f32 / b as f32).build();
                        }
                        _ => {
                            ui.progress_bar(0f32).build();
                        }
                    }
                })
        });
    }
}
