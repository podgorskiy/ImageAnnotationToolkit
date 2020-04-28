/*
 * Copyright 2017-2020 Stanislav Pidhorskyi. All rights reserved.
 * License: https://raw.githubusercontent.com/podgorskiy/bimpy/master/LICENSE.txt
 */
#include "Camera2D.h"
#include "DebugRenderer.h"
#include "Shader.h"
#include "VertexSpec.h"
#include "VertexBuffer.h"
#include <glm/ext/matrix_transform.hpp>
#include "Vector/nanovg.h"
#include "Vector/nanovg_backend.h"
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


class Image
{
public:
	Image& operator=(const Image&) = delete;
	Image(const Image&) = delete;

	Image(ndarray_uint8 ndarray): Image(ndarray.request())
	{}

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
private:
	Image(const py::buffer_info& ndarray_info)
	{
		glGenTextures(1, &m_textureHandle);
		glBindTexture(GL_TEXTURE_2D, m_textureHandle);

		m_width = -1;
		m_height = -1;

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
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, m_width, m_height, 0, GL_RED, GL_UNSIGNED_BYTE, ndarray_info.ptr);
		}
		else if (ndarray_info.ndim == 3)
		{
			m_width = ndarray_info.shape[1];
			m_height = ndarray_info.shape[0];

			if (ndarray_info.shape[2] == 1)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_R);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, ndarray_info.ptr);
			}
			else if (ndarray_info.shape[2] == 2)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_RG);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, m_width, m_height, 0, GL_RG, GL_UNSIGNED_BYTE, ndarray_info.ptr);
			}
			else if (ndarray_info.shape[2] == 3)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_RGB);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, ndarray_info.ptr);
			}
			else if (ndarray_info.shape[2] == 4)
			{
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask_RGBA);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ndarray_info.ptr);
			}
			else
			{
				glBindTexture(GL_TEXTURE_2D, 0);
				glPixelStorei(GL_UNPACK_ALIGNMENT, backup);
				throw std::runtime_error("Wrong number of channels. Should be either 1, 2, 3, or 4");
			}
		}
		else
		{
			glBindTexture(GL_TEXTURE_2D, 0);
			glPixelStorei(GL_UNPACK_ALIGNMENT, backup);
			throw std::runtime_error("Wrong number of dimensions. Should be either 2 or 3");
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);

		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glPixelStorei(GL_UNPACK_ALIGNMENT, backup);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	GLuint m_textureHandle;
};

typedef std::shared_ptr<Image> ImagePtr;


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

	void NewFrame();

	void Render();

	bool ShouldClose();

	int GetWidth() const;

	int GetHeight() const;

	~Context();

	GLFWwindow* m_window = nullptr;
	int m_width;
	int m_height;
	py::function mouse_button_callback;
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
		glfwInit();

#if __APPLE__
		// GL 3.2 + GLSL 150
		const char* glsl_version = "#version 150";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
		// GL 3.0 + GLSL 130
		const char* glsl_version = "#version 130";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		glfwWindowHint(GLFW_SRGB_CAPABLE, 1);

		//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

		m_window = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);

		glfwMakeContextCurrent(m_window);
		gl3wInit();
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

		glfwSetKeyCallback(m_window, [](GLFWwindow*, int key, int, int action, int mods)
		{
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
				vec4 color = texture2D(u_texture, q);
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
	}
}


void Context::Recenter(RECENTER r)
{
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


Context::~Context()
{
	glfwSetWindowSizeCallback(m_window, nullptr);
	glfwTerminate();
}


void Context::Render()
{
	glfwMakeContextCurrent(m_window);
	glViewport(0, 0, m_width, m_height);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_FRAMEBUFFER_SRGB);

	double x, y;
	glfwGetCursorPos(m_window, &x, &y);
	glm::vec2 cursorposition = glm::vec2(x, y);
	m_camera.Move(cursorposition.x, cursorposition.y);

	m_camera.UpdateViewProjection(m_width, m_height);

	auto size = m_image->GetSize();

	auto transform = m_camera.GetTransform();
	glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(size.x / 2.0f,  size.y / 2.0f, 0.0f)) * glm::translate(glm::mat4(1.0f), glm::vec3(1.0, 1.0, 0.0f));
	// Render::DrawRect(m_dr, glm::vec2(-1.0f), glm::vec2(1.0f), transform * model);
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
		nvgBeginFrame(vg, m_width, m_height, 1.0f);
		auto transform = m_camera.GetCanvasToWorld();

		glm::vec2 pos = transform * glm::vec3(0, 0, 1);
		size = transform * glm::vec3(size, 0);


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

	glfwSwapInterval(1);
	glfwSwapBuffers(m_window);
	glfwPollEvents();
}


void Context::NewFrame()
{
}


void Context::Resize(int width, int height)
{
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
		.def("__enter__", &Context::NewFrame)
		.def("__exit__", [](Context& self, py::object, py::object, py::object)
			{
				self.Render();
			})
		.def("set_mouse_button_callback", [](Context& self, py::function f){
			self.mouse_button_callback = f;
		})
		.def("set_keyboard_callback", [](Context& self, py::function f){
			self.keyboard_callback = f;
		});

	py::class_<Image, std::shared_ptr<Image> >(m, "Image")
			.def(py::init<ndarray_uint8>(), "Constructs Image object from ndarray, PIL Image, numpy array, etc.")
			.def("grayscale_to_alpha", &Image::GrayScaleToAlpha, "For grayscale images, uses values as alpha")
			.def_readonly("width", &Image::m_width)
			.def_readonly("height", &Image::m_height);

}