/*
 * Copyright 2017-2020 Stanislav Pidhorskyi. All rights reserved.
 * License: https://raw.githubusercontent.com/podgorskiy/bimpy/master/LICENSE.txt
 */
#include "Camera2D.h"
#include "DebugRenderer.h"
#include "simpletext.h"
#include "GLDebugMessage.h"
#include "Shader.h"
#include "VertexSpec.h"
#include "VertexBuffer.h"
#include <glm/ext/matrix_transform.hpp>
#include "Vector/nanovg.h"
#include "Vector/nanovg_backend.h"
#include "runtime_error.h"
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <mutex>
#include <spdlog/spdlog.h>

namespace py = pybind11;

typedef py::array_t<uint8_t, py::array::c_style> ndarray_uint8;

enum SpecialKeys
{
	KeyEscape = 256,
	KeyEnter = 257,
	KeyTab = 258,
	KeyBackspace = 259,
	KeyInsert = 260,
	KeyDelete = 261,
	KeyRight = 262,
	KeyLeft = 263,
	KeyDown = 264,
	KeyUp = 265,
};


class Image
{
public:
	Image& operator=(const Image&) = delete;
	Image(const Image&) = delete;

	Image()
	{
		glGenTextures(1, &m_textureHandle);
		m_width = -1;
		m_height = -1;
	}

	Image(std::vector<ndarray_uint8> ims)
	{
		glGenTextures(1, &m_textureHandle);
		m_width = -1;
		m_height = -1;
		SetImage(ims);
	}

	~Image()
	{
		glDeleteTextures(1, &m_textureHandle);
	}

	GLuint GetHandle() const
	{
		return m_textureHandle;
	}

	void GrayScaleToAlpha()
	{
		GLint swizzleMask[] = { GL_ONE, GL_ONE, GL_ONE, GL_RED };
		glBindTexture(GL_TEXTURE_2D, m_textureHandle);
		glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glm::vec2 GetSize() const
	{
		return glm::vec2(m_width, m_height);
	}

	ssize_t m_width;
	ssize_t m_height;

	void SetImage(std::vector<ndarray_uint8> ims)
	{
		// Render::debug_guard<> m_guard;
		const py::buffer_info& ndarray_info = ims[0].request();
		glBindTexture(GL_TEXTURE_2D, m_textureHandle);

		GLint backup;
		glGetIntegerv(GL_UNPACK_ALIGNMENT, &backup);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		GLint swizzleMask_R[] = { GL_RED, GL_RED, GL_RED, GL_ONE };
		GLint swizzleMask_RG[] = { GL_RED, GL_GREEN, GL_ZERO, GL_ONE };
		GLint swizzleMask_RGB[] = { GL_RED, GL_GREEN, GL_BLUE, GL_ONE };
		GLint swizzleMask_RGBA[] = { GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA };

		if (ndarray_info.ndim == 2)
		{
			m_width = ndarray_info.shape[1];
			m_height = ndarray_info.shape[0];
			glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_R);
			int mipmap = 0;
			for (auto im: ims)
			{
				glTexImage2D(GL_TEXTURE_2D, mipmap, GL_R8, im.request().shape[1], im.request().shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, im.request().ptr);
				mipmap += 1;
			}
		}
		else if (ndarray_info.ndim == 3)
		{
			m_width = ndarray_info.shape[1];
			m_height = ndarray_info.shape[0];
			if (ndarray_info.shape[2] == 1)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_R);
				int mipmap = 0;
				for (auto im: ims)
				{
					glTexImage2D(GL_TEXTURE_2D, mipmap, GL_R8, im.request().shape[1], im.request().shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, im.request().ptr);
					mipmap += 1;
				}
			}
			else if (ndarray_info.shape[2] == 2)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_RG);
				int mipmap = 0;
				for (auto im: ims)
				{
					glTexImage2D(GL_TEXTURE_2D, mipmap, GL_RG8, im.request().shape[1], im.request().shape[0], 0, GL_RG, GL_UNSIGNED_BYTE, im.request().ptr);
					mipmap += 1;
				}
			}
			else if (ndarray_info.shape[2] == 3)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_RGB);
				int mipmap = 0;
				for (auto im: ims)
				{
					glTexImage2D(GL_TEXTURE_2D, mipmap, GL_SRGB8, im.request().shape[1], im.request().shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, im.request().ptr);
					mipmap += 1;
				}
			}
			else if (ndarray_info.shape[2] == 4)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_RGBA);
				int mipmap = 0;
				for (auto im: ims)
				{
					glTexImage2D(GL_TEXTURE_2D, mipmap, GL_SRGB8_ALPHA8, im.request().shape[1], im.request().shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, im.request().ptr);
					mipmap += 1;
				}
			}
			else
			{
				glBindTexture(GL_TEXTURE_2D, 0);
				glPixelStorei(GL_UNPACK_ALIGNMENT, backup);
				throw runtime_error("Wrong number of channels. Should be either 1, 2, 3, or 4, but got %d", (int)ndarray_info.shape[2]);
			}
		}
		else
		{
			glBindTexture(GL_TEXTURE_2D, 0);
			glPixelStorei(GL_UNPACK_ALIGNMENT, backup);
			throw runtime_error("Wrong number of dimensions. Should be either 2 or 3, but got %d", (int)ndarray_info.ndim);
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		if (ims.size() > 1)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		else
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glPixelStorei(GL_UNPACK_ALIGNMENT, backup);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	GLuint m_textureHandle;
};

typedef std::shared_ptr<Image> ImagePtr;
typedef std::shared_ptr<SimpleText> SimpleTextPtr;

class Context
{
public:
	enum RECENTER
	{
		FIT_DOCUMENT,
		ORIGINAL_SIZE
	};

	Context& operator=(const Context&) = delete;
	Context(const Context&) = delete;
	Context() = default;

	void Init(int width, int height, const std::string& name);

	void Resize(int width, int height);

	void Recenter(RECENTER r);
	void Recenter(float x0, float y0, float x1, float y1);

	void NewFrame();

	void Render();

	bool ShouldClose();

	int GetWidth() const;

	int GetHeight() const;

	void Point(float x, float y, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color, float point_size) const;
	void Box(float minx, float miny, float maxx, float maxy, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color_stroke, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color_fill) const;

	~Context();

	GLFWwindow* m_window = nullptr;
	int m_width;
	int m_height;
	py::function mouse_button_callback;
	py::function mouse_position_callback;
	py::function keyboard_callback;

	Camera2D m_camera;
	Render::DebugRenderer m_dr;
	ImagePtr m_image;
	NVGcontext* vg = nullptr;
	Render::VertexSpec m_spec;
	Render::VertexBuffer m_buff;
	Render::ProgramPtr m_program;
	Render::Uniform u_modelViewProj;
	Render::Uniform u_texture;
	SimpleTextPtr m_text;
};

struct Vertex
{
	glm::vec2 pos;
	glm::vec2 uv;
};

void Context::Init(int width, int height, const std::string& name)
{
	if (nullptr == m_window)
	{
		if (!glfwInit())
		{
			throw runtime_error("GLFW initialization failed.\nThis may happen if you try to run bimpy on a headless machine ");
		}

		glfwWindowHint(GLFW_DEPTH_BITS, 24);
		glfwWindowHint(GLFW_RED_BITS, 8);
#if __APPLE__
		// GL 3.2 + GLSL 150
		// const char* glsl_version = "#version 150";
		// glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		// glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
		// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
		// GL 3.0 + GLSL 130
		const char* glsl_version = "#version 130";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

		//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif
		glfwWindowHint(GLFW_SRGB_CAPABLE, 1);

		m_window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
	    if (!m_window)
	    {
	        glfwTerminate();
			throw runtime_error("GLFW failed to create window.\nThis may happen if you try to run bimpy on a headless machine ");
	    }

		glfwMakeContextCurrent(m_window);
		if (gl3wInit() != GL3W_OK)
		{
			// throw runtime_error("GL3W initialization failed.\nThis may happen if you try to run bimpy on a headless machine ");
		}

		// Render::debug_guard<> m_guard;
		m_dr.Init();

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

		m_width = width;
		m_height = height;

		glfwSetWindowUserPointer(m_window, this); // replaced m_imp.get()

		glfwSetWindowSizeCallback(m_window, [](GLFWwindow* window, int width, int height)
		{
			Context* ctx = static_cast<Context*>(glfwGetWindowUserPointer(window));
			ctx->Resize(width, height);
		});

		glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int, int action, int mods)
		{
			Context* ctx = static_cast<Context*>(glfwGetWindowUserPointer(window));

			ctx->keyboard_callback(key, action, mods);
		});

		glfwSetCharCallback(m_window, [](GLFWwindow*, unsigned int c)
		{
		});

		glfwSetScrollCallback(m_window, [](GLFWwindow* window, double /*xoffset*/, double yoffset)
		{
			Context* ctx = static_cast<Context*>(glfwGetWindowUserPointer(window));

			ctx->m_camera.Scroll(float(-yoffset));
		});

		glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int /*mods*/)
		{
			Context* ctx = static_cast<Context*>(glfwGetWindowUserPointer(window));
			if (button == 1)
				ctx->m_camera.TogglePanning(action == GLFW_PRESS);
			if (button == 0 && ctx->mouse_button_callback)
			{
				double x, y;
				glfwGetCursorPos(window, &x, &y);
				glm::vec2 cursorposition = glm::vec2(x, y);
				auto local = glm::vec2(ctx->m_camera.GetWorldToCanvas() * glm::vec3(cursorposition, 1.0f));
				ctx->mouse_button_callback(action == GLFW_PRESS, float(x), float(y), local.x, local.y);
			}
		});
		glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double x, double y)
		{
			Context* ctx = static_cast<Context*>(glfwGetWindowUserPointer(window));
			if (ctx->mouse_position_callback)
			{
				glm::vec2 cursorposition = glm::vec2(x, y);
				auto local = glm::vec2(ctx->m_camera.GetWorldToCanvas() * glm::vec3(cursorposition, 1.0f));
				ctx->mouse_position_callback(float(x), float(y), local.x, local.y);
			}
		});

		vg = nvgCreateContext(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
		if (vg == nullptr)
		{
			spdlog::error("Error, Could not init nanovg.");
		}

		const char* vertex_shader_src = R"(
			uniform mat4  u_modelViewProj;

			attribute vec2 a_position;
			varying vec2 v_pos;

			void main()
			{
				v_pos = a_position.xy;
				gl_Position = u_modelViewProj * vec4(a_position, 0.0, 1.0);
			}
		)";

		const char* fragment_shader_src = R"(
			uniform sampler2D u_texture;
			varying vec2 v_pos;

			vec3 sample(vec2 q)
			{
				vec4 color = texture2D(u_texture, q, -0.3);
				return color.rgb;
			}

			void main()
			{
				vec2 q = v_pos * vec2(0.5, 0.5);
			    vec3 color = sample(vec2(0.5) + q).rgb;
				gl_FragColor = vec4(color, 1.0);
			}
		)";

		m_program = Render::MakeProgram(vertex_shader_src, fragment_shader_src);
		u_modelViewProj = m_program->GetUniform("u_modelViewProj");
		u_texture = m_program->GetUniform("u_texture");
		std::vector<glm::vec2> vertices = {
				{-1.0f, -1.0f},
				{ 1.0f, -1.0f},
				{ 1.0f,  1.0f},
				{-1.0f,  1.0f},
		};
		std::vector<int> indices;

		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);
		indices.push_back(0);
		indices.push_back(2);
		indices.push_back(3);

		m_buff.FillBuffers(vertices.data(), vertices.size(), sizeof(glm::vec2), indices.data(), indices.size(), 4);

		m_spec = Render::VertexSpecMaker().PushType<glm::vec2>("a_position");

		m_text.reset(new SimpleText);
	}
}


void Context::Recenter(RECENTER r)
{
	if (!m_image)
	{
		throw std::runtime_error("No image assigned");
	}
	auto size = m_image->GetSize();
	if (r == RECENTER::FIT_DOCUMENT)
	{
		glm::vec2 r = glm::vec2(m_width, m_height) / glm::vec2(size);

		if (m_width * 1.0f / m_height > size.x * 1.0f / size.y)
		{
			m_camera.SetFOV(size.y * 1.2f / m_height);
		}
		else
		{
			m_camera.SetFOV(size.x * 1.2f / m_width);
		}
	}
	else
	{
		m_camera.SetFOV(1.0f);
	}

	auto clientArea = glm::vec2(m_width, m_height);

	clientArea = glm::ivec2(glm::vec2(clientArea) * m_camera.GetFOV());

	auto pos = glm::vec2(clientArea - size) / 2.0f;
	m_camera.SetPos(pos);
}


void Context::Recenter(float x0, float y0, float x1, float y1)
{
	if (!m_image)
	{
		throw std::runtime_error("No image assigned");
	}
	auto p0 = glm::vec2(x0, y0);
	auto p1 = glm::vec2(x1, y1);
	auto size = p1 - p0;

	glm::vec2 r = glm::vec2(m_width, m_height) / glm::vec2(size);

	if (m_width * 1.0f / m_height > size.x * 1.0f / size.y)
	{
		m_camera.SetFOV(size.y * 1.2f / m_height);
	}
	else
	{
		m_camera.SetFOV(size.x * 1.2f / m_width);
	}

	auto clientArea = glm::vec2(m_width, m_height);

	clientArea = glm::ivec2(glm::vec2(clientArea) * m_camera.GetFOV());

	auto pos = glm::vec2(clientArea - size) / 2.0f;
	m_camera.SetPos(pos - p0);
}


Context::~Context()
{
	glfwSetWindowSizeCallback(m_window, nullptr);
	glfwTerminate();
}


void Context::Render()
{
	auto size = m_image->GetSize();

	auto transform = m_camera.GetTransform();
	glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3((size.x + 1) / 2.0f,  (size.y + 1) / 2.0f, 0.0f)) * glm::translate(glm::mat4(1.0f), glm::vec3(1.0, 1.0, 0.0f));
	model[3].x -= 0.5;
	model[3].y -= 0.5;
	// Render::DrawRect(m_dr, glm::vec2(-1.0f), glm::vec&Image::2(1.0f), transform * model);
	{
		m_program->Use();
		u_modelViewProj.ApplyValue(transform * model);
		u_texture.ApplyValue(0);
		glBindTexture(GL_TEXTURE_2D, m_image->GetHandle());

		m_buff.Bind();
		m_spec.Enable();
		m_buff.DrawElements();
		m_buff.UnBind();
		m_spec.Disable();

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	{
		auto transform = m_camera.GetCanvasToWorld();

		glm::vec2 pos = transform * glm::vec3(-0.5, -0.5, 1);
		size = transform * glm::vec3(size + 1.0f, 0);


		float margin = size.x * 0.3;
		NVGpaint shadowPaint = nvgBoxGradient(
				vg, pos.x, pos.y, size.x, size.y, 0, margin * 0.03,
				{0, 0, 0, 1.0f}, {0, 0, 0, 0});

		nvgSave(vg);
		nvgResetScissor(vg);
		nvgBeginPath(vg);
		nvgRect(vg, pos.x - margin, pos.y - margin, size.x + 2 * margin, size.y + 2 * margin);
		nvgRect(vg, pos.x, pos.y, size.x, size.y);
		nvgPathWinding(vg, NVG_HOLE);
		nvgFillPaint(vg, shadowPaint);
		nvgFill(vg);
		nvgRestore(vg);
		nvgEndFrame(vg);
	}

	m_text->EnableBlending(true);
	m_text->Render();

	glfwSwapInterval(1);
	glfwSwapBuffers(m_window);
	glfwPollEvents();
}


void Context::NewFrame()
{
	double x, y;
	glfwGetCursorPos(m_window, &x, &y);
	glm::vec2 cursorposition = glm::vec2(x, y);
	m_camera.Move(cursorposition.x, cursorposition.y);
	m_camera.UpdateViewProjection(m_width, m_height);

	if (!m_image)
	{
		throw std::runtime_error("No image assigned");
	}
	glfwMakeContextCurrent(m_window);
	Render::debug_guard<> m_guard;
	glViewport(0, 0, m_width, m_height);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_FRAMEBUFFER_SRGB);
	nvgBeginFrame(vg, m_width, m_height, 1.0f);
}


void Context::Resize(int width, int height)
{
	if (!m_image)
	{
		throw std::runtime_error("No image assigned");
	}
	auto oldWindowBufferSize = glm::vec2(m_width, m_height);
	m_width = width;
	m_height = height;
	m_camera.UpdateViewProjection(m_width, m_height);
	auto size = m_image->GetSize();

	auto oldClientArea = oldWindowBufferSize;
	auto clientArea = glm::vec2(m_width, m_height);// - glm::ivec2(0, MainMenuBar * m_window->GetPixelScale());

	auto pos = m_camera.GetPos();
	auto delta = glm::vec2((clientArea - oldClientArea) / 2.0f) * m_camera.GetFOV();
	m_camera.SetPos(pos + delta);
}


bool Context::ShouldClose()
{
	return glfwWindowShouldClose(m_window) != 0;
}

int Context::GetWidth() const
{
	return m_width;
}

int Context::GetHeight() const
{
	return m_height;
}

void Context::Point(float x, float y, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color, float point_size) const
{
	auto transform = m_camera.GetCanvasToWorld();

	glm::vec2 point_pos_local = glm::vec2(x, y);
	glm::vec2 point_pos = transform * glm::vec3(point_pos_local, 1);

	nvgBeginPath(vg);
	nvgCircle(vg, point_pos.x, point_pos.y, point_size);
	nvgFillColor(vg, nvgRGBA(std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color)));
	nvgFill(vg);
	nvgBeginPath(vg);

	nvgCircle(vg, point_pos.x, point_pos.y, point_size * 6);
	nvgCircle(vg, point_pos.x, point_pos.y, point_size);
	nvgPathWinding(vg, NVG_HOLE);
	NVGpaint rshadowPaint = nvgBoxGradient(
			vg, point_pos.x - point_size, point_pos.y - point_size, point_size * 2, point_size * 2, point_size,
			point_size * 0.3,
			{0, 0, 0, 1.0f}, {0, 0, 0, 0});
	nvgFillPaint(vg, rshadowPaint);
	nvgFill(vg);
}

void Context::Box(float minx, float miny, float maxx, float maxy, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color_stroke, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color_fill) const
{
	auto transform = m_camera.GetCanvasToWorld();

	glm::aabb2 box(transform * glm::vec3(minx, miny, 1), transform * glm::vec3(maxx, maxy, 1));

	nvgBeginPath(vg);
	nvgRect(vg, box);
	nvgFillColor(vg, nvgRGBA(std::get<0>(color_stroke), std::get<1>(color_fill), std::get<2>(color_fill), std::get<3>(color_fill)));
	nvgFill(vg);

	nvgBeginPath(vg);
	nvgRect(vg, box);
	nvgStrokeColor(vg, nvgRGBA(std::get<0>(color_stroke), std::get<1>(color_stroke), std::get<2>(color_stroke), std::get<3>(color_stroke)));
	nvgStroke(vg);
}


PYBIND11_MODULE(_anntoolkit, m) {
	m.doc() = "anntoolkit";

	py::class_<Context>(m, "Context")
		.def(py::init())
		.def("init", &Context::Init, "Initializes context and creates window")
		.def("new_frame", &Context::NewFrame, "Starts a new frame. NewFrame must be called before any imgui functions")
		.def("render", &Context::Render, "Finilizes the frame and draws all UI. Render must be called after all imgui functions")
		.def("should_close", &Context::ShouldClose)
		.def("width", &Context::GetWidth)
		.def("height", &Context::GetHeight)
		.def("set", [](Context& self, ImagePtr im)
			{
				self.m_image = im;
				self.Recenter(Context::FIT_DOCUMENT);
			})
		.def("set_without_recenter", [](Context& self, ImagePtr im)
			{
				self.m_image = im;
			})
		.def("recenter", [](Context& self)
			{
				self.Recenter(Context::FIT_DOCUMENT);
			})
		.def("set_roi", [](Context& self, float x0, float y0, float x1, float y1)
			{
				self.Recenter(x0, y0, x1, y1);
			})
		.def("__enter__", &Context::NewFrame)
		.def("__exit__", [](Context& self, py::object, py::object, py::object)
			{
				self.Render();
			})
		.def("set_mouse_button_callback", [](Context& self, py::function f){
			self.mouse_button_callback = f;
		})
		.def("set_mouse_button_callback", [](Context& self, py::function f){
			self.mouse_button_callback = f;
		})
		.def("set_mouse_position_callback", [](Context& self, py::function f){
			self.mouse_position_callback = f;
		})
		.def("get_mouse_position", [](Context& self){
				double x, y;
				glfwGetCursorPos(self.m_window, &x, &y);
				glm::vec2 cursorposition = glm::vec2(x, y);
				auto local = glm::vec2(self.m_camera.GetWorldToCanvas() * glm::vec3(cursorposition, 1.0f));
				return std::make_tuple(float(x), float(y), local.x, local.y);
		})
		.def("set_keyboard_callback", [](Context& self, py::function f){
			self.keyboard_callback = f;
		})
		.def("text", [](Context& self, const char* str, int x, int y, SimpleText::Alignment align)
		{
			self.m_text->Label(str, x, y, align);
		})
		.def("text", [](Context& self, const char* str, int x, int y, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> bg_color, SimpleText::Alignment align)
		{
			self.m_text->SetColorf(SimpleText::TEXT_COLOR, std::get<0>(color) / 255.f, std::get<1>(color) / 255.f, std::get<2>(color) / 255.f, std::get<3>(color) / 255.f);
			self.m_text->SetColorf(SimpleText::BACKGROUND_COLOR, std::get<0>(bg_color) / 255.f, std::get<1>(bg_color) / 255.f, std::get<2>(bg_color) / 255.f, std::get<3>(bg_color) / 255.f);
			self.m_text->EnableBlending(true);
			self.m_text->Label(str, x, y, align);
			self.m_text->ResetFont();
		})
		.def("text_loc", [](Context& self, const char* str, float x, float y, SimpleText::Alignment align)
		{
			auto transform = self.m_camera.GetCanvasToWorld();

			glm::vec2 pos_local = glm::vec2(x, y);
			glm::vec2 pos = transform * glm::vec3(pos_local, 1);

			self.m_text->Label(str, pos.x, pos.y, align);
		})
		.def("loc_2_win", [](Context& self, float x, float y)
		{
			auto transform = self.m_camera.GetCanvasToWorld();
			glm::vec2 pos_local = glm::vec2(x, y);
			glm::vec2 pos = transform * glm::vec3(pos_local, 1);
			return std::tuple<float, float>(pos.x, pos.y);
		})
		.def("win_2_loc", [](Context& self, float x, float y)
		{
			auto transform = self.m_camera.GetWorldToCanvas();
			glm::vec2 pos_local = glm::vec2(x, y);
			glm::vec2 pos = transform * glm::vec3(pos_local, 1);
			return std::tuple<float, float>(pos.x, pos.y);
		})
		.def("get_scale", [] (Context& self) { return 1.0 / self.m_camera.GetFOV(); })
		.def("text_loc", [](Context& self, const char* str, float x, float y, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> color, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t> bg_color, SimpleText::Alignment align)
		{
			auto transform = self.m_camera.GetCanvasToWorld();

			glm::vec2 pos_local = glm::vec2(x, y);
			glm::vec2 pos = transform * glm::vec3(pos_local, 1);

			self.m_text->SetColorf(SimpleText::TEXT_COLOR, std::get<0>(color) / 255.f, std::get<1>(color) / 255.f, std::get<2>(color) / 255.f, std::get<3>(color) / 255.f);
			self.m_text->SetColorf(SimpleText::BACKGROUND_COLOR, std::get<0>(bg_color) / 255.f, std::get<1>(bg_color) / 255.f, std::get<2>(bg_color) / 255.f, std::get<3>(bg_color) / 255.f);
			self.m_text->EnableBlending(true);

			self.m_text->Label(str, pos.x, pos.y, align);
			self.m_text->ResetFont();
		})
		.def("point",  &Context::Point, py::arg("x"), py::arg("y"), py::arg("color"), py::arg("radius") = 5)
		.def("box",  &Context::Box);

		py::enum_<SpecialKeys>(m, "SpecialKeys")
			.value("KeyEscape", KeyEscape)
			.value("KeyEnter", KeyEnter)
			.value("KeyTab", KeyTab)
			.value("KeyBackspace", KeyBackspace)
			.value("KeyInsert", KeyInsert)
			.value("KeyDelete", KeyDelete)
			.value("KeyRight", KeyRight)
			.value("KeyLeft", KeyLeft)
			.value("KeyDown", KeyDown)
			.value("KeyUp", KeyUp)
			.export_values();

		py::enum_<SimpleText::Alignment>(m, "Alignment")
			.value("Left", SimpleText::LEFT)
			.value("Center", SimpleText::CENTER)
			.value("Right", SimpleText::RIGHT)
			.export_values();

	py::class_<Image, std::shared_ptr<Image> >(m, "Image")
			.def(py::init<std::vector<ndarray_uint8>>(), "")
			.def("grayscale_to_alpha", &Image::GrayScaleToAlpha, "For grayscale images, uses values as alpha")
			.def_readonly("width", &Image::m_width)
			.def_readonly("height", &Image::m_height);
}
