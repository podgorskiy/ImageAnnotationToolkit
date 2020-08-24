#include "Shader.h"
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <GL/gl3w.h>


namespace Render
{

	Shader::Shader(SHADER_TYPE::Type t): m_type(t)
	{
		GLuint type;
		switch (t)
		{
			case SHADER_TYPE::COMPUTE_SHADER:
				type = GL_COMPUTE_SHADER;
				break;
			case SHADER_TYPE::GEOMETRY_SHADER:
				type = GL_GEOMETRY_SHADER;
				break;
			case SHADER_TYPE::VERTEX_SHADER:
				type = GL_VERTEX_SHADER;
				break;
			case SHADER_TYPE::FRAGMENT_SHADER:
				type = GL_FRAGMENT_SHADER;
				break;
		}
		m_shader = glCreateShader(type);
	}

	std::string GetShaderTypeName(SHADER_TYPE::Type t)
	{
		switch (t)
		{
			case SHADER_TYPE::COMPUTE_SHADER:
				return "COMPUTE_SHADER";
			case SHADER_TYPE::GEOMETRY_SHADER:
				return "GEOMETRY_SHADER";
			case SHADER_TYPE::VERTEX_SHADER:
				return "VERTEX_SHADER";
			case SHADER_TYPE::FRAGMENT_SHADER:
				return "FRAGMENT_SHADER";
		}
		return "INVALID_SHADER";
	}

	Shader::~Shader()
	{
		glDeleteShader(m_shader);
	}

	static void PrintSource(const char* source, int startLine)
	{
		int line = startLine;
		while (true)
		{
			const char* start = source;
			while (*source != '\n' && *source != '\0')
			{
				++source;
			}
			std::string l(start, source);
			spdlog::info("{:3d}:\t{}", line++, l);

			if (*source == '\0')
			{
				break;
			}
			++source;
		}
	}

	bool Shader::CompileShader(const char* src)
	{
		glShaderSource(m_shader, 1, &src, NULL);

		glCompileShader(m_shader);
		GLint compiled = 0;
		glGetShaderiv(m_shader, GL_COMPILE_STATUS, &compiled);
		GLint infoLen = 0;
		glGetShaderiv(m_shader, GL_INFO_LOG_LENGTH, &infoLen);

		if (infoLen > 1)
		{
			spdlog::warn("{} during {} shader compilation.", compiled == GL_TRUE ? "Warning" : "Error",
			             GetShaderTypeName(m_type));

			PrintSource(src, 1);

			char* buf = new char[infoLen];
			glGetShaderInfoLog(m_shader, infoLen, NULL, buf);
			spdlog::warn("Compilation log: {}", buf);
			delete[] buf;
		}
		return compiled == GL_TRUE;
	}
}
