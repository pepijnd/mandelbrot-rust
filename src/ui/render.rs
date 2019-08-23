use glium::{
    backend::Facade,
    index::PrimitiveType,
    texture::{RawImage2d, Texture2d},
    Surface,
};

use crate::mandelbrot::{bounded::Bound, compute::ComputedSet};

use crate::ui::app::AppState;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}
implement_vertex!(Vertex, position, tex_coords);

pub struct AppRenderer {
    computed_set_tex_cache: Option<Texture2d>,
}

impl AppRenderer {
    pub fn init() -> AppRenderer {
        AppRenderer {
            computed_set_tex_cache: None,
        }
    }

    pub fn render<T, F>(&mut self, state: &mut AppState, target: &mut T, facade: &F)
    where
        T: Surface,
        F: Facade,
    {
        if !state.set_valid || self.computed_set_tex_cache.is_none() {
            self.computed_set_tex_cache = Some(state.computed_set.make_texture(facade));
            state.set_valid = true;
        }
        AppRenderer::render_texture(
            self.computed_set_tex_cache.as_ref().unwrap(),
            target,
            facade,
        );
        if state.dragging {
            AppRenderer::render_select(target, facade, state);
        }
    }

    fn render_texture<T, F>(tex: &Texture2d, target: &mut T, facade: &F)
    where
        T: Surface,
        F: Facade,
    {
        let vertex_buffer = {
            glium::VertexBuffer::new(
                facade,
                &[
                    Vertex {
                        position: [-1.0, -1.0],
                        tex_coords: [0.0, 0.0],
                    },
                    Vertex {
                        position: [-1.0, 1.0],
                        tex_coords: [0.0, 1.0],
                    },
                    Vertex {
                        position: [1.0, 1.0],
                        tex_coords: [1.0, 1.0],
                    },
                    Vertex {
                        position: [1.0, -1.0],
                        tex_coords: [1.0, 0.0],
                    },
                ],
            )
            .unwrap()
        };

        let index_buffer =
            glium::IndexBuffer::new(facade, PrimitiveType::TriangleStrip, &[1 as u16, 2, 0, 3])
                .unwrap();

        let program = program!(facade,
            140 => {
                vertex: "
                #version 140
                uniform mat4 matrix;
                in vec2 position;
                in vec2 tex_coords;
                out vec2 v_tex_coords;
                void main() {
                    gl_Position = matrix * vec4(position, 0.0, 1.0);
                    v_tex_coords = tex_coords;
                }
            ",

                fragment: "
                #version 140
                uniform sampler2D tex;
                in vec2 v_tex_coords;
                out vec4 f_color;
                void main() {
                    f_color = texture(tex, v_tex_coords);
                }
            "
            },
        )
        .unwrap();

        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32]
            ],
            tex: tex
        };
        target
            .draw(
                &vertex_buffer,
                &index_buffer,
                &program,
                &uniforms,
                &Default::default(),
            )
            .unwrap();
    }

    fn render_select<T, F>(target: &mut T, facade: &F, state: &AppState)
    where
        T: Surface,
        F: Facade,
    {
        let (x1, y1, x2, y2) = (
            2.0 * state.mouse_start[0] as f32 - 1.0,
            -2.0 * state.mouse_start[1] as f32 + 1.0,
            2.0 * state.mouse_end[0] as f32 - 1.0,
            -2.0 * state.mouse_end[1] as f32 + 1.0,
        );

        let rect = glium::VertexBuffer::new(
            facade,
            &[
                Vertex {
                    position: [x1, y1],
                    tex_coords: [0.0, 0.0],
                },
                Vertex {
                    position: [x2, y1],
                    tex_coords: [0.0, 0.0],
                },
                Vertex {
                    position: [x2, y2],
                    tex_coords: [0.0, 0.0],
                },
                Vertex {
                    position: [x1, y2],
                    tex_coords: [0.0, 0.0],
                },
            ],
        )
        .unwrap();

        let index_buffer =
            glium::IndexBuffer::new(facade, PrimitiveType::LineLoop, &[0 as u16, 1, 2, 3]).unwrap();

        let uniforms = uniform! {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32]
            ]
        };

        let program = program!(facade, 140 => {
            vertex: "
                #version 140

                uniform mat4 matrix;
                in vec2 position;
                in vec2 tex_coords;
                void main() {
                    gl_Position = matrix * vec4(position, 0.0, 1.0);
                }
            ",
            fragment: "
                #version 140
                
                out vec4 color;
                void main() {
                    color = vec4(1.0, 0.0, 0.0, 0.0);
                }
            "
        })
        .unwrap();

        target
            .draw(
                &rect,
                &index_buffer,
                &program,
                &uniforms,
                &Default::default(),
            )
            .unwrap();
    }
}

pub trait MakeTexture<F>
where
    F: Facade,
{
    fn make_texture(&self, facade: &F) -> Texture2d;
}

impl<F> MakeTexture<F> for ComputedSet
where
    F: Facade,
{
    fn make_texture(&self, facade: &F) -> Texture2d {
        match self.iter() {
            Some(data) => Texture2d::new(
                facade,
                RawImage2d::from_raw_rgba(
                    data.flat_map(|bound| match bound {
                        Bound::Bounded => vec![0.0, 0.0, 0.0, 1.0],
                        Bound::Unbounded(n) => vec![*n as f32 / 500.0, 0.0, 0.0, 1.0],
                    })
                    .collect::<Vec<f32>>(),
                    self.get_size(),
                ),
            )
            .unwrap(),
            None => Texture2d::empty(facade, self.get_size().0, self.get_size().1).unwrap(),
        }
    }
}
