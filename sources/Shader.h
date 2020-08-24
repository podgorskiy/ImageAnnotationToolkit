#pragma once
#include "Types.h"
#include <vector>
#include <array>
#include <map>
#include <memory>


namespace Render
{
	typedef unsigned int GLHandle;
	struct SHADER_TYPE
	{
		enum Type
		{
			VERTEX_SHADER,
			GEOMETRY_SHADER,
			FRAGMENT_SHADER,
			COMPUTE_SHADER,
		};
	};

	class Shader
	{
		Shader(const Shader& other) = delete;
		Shader& operator=(const Shader&) = delete;
		friend class Program;
	public:
		Shader(SHADER_TYPE::Type type);
		~Shader();

		bool CompileShader(const char* src);

	private:
		SHADER_TYPE::Type m_type;
		GLHandle m_shader;
	};


	class Uniform
	{
	public:
		Uniform(): m_handle(-1), m_type(VarType::INVALID), m_num(0) {}
		Uniform(int handle, VarType::Type type, int num): m_handle(handle), m_type(type), m_num(num) {};

		Uniform(const Uniform& other) = default;
		Uniform& operator=(const Uniform&) = default;

		bool valid() const { return m_handle != -1; }

		VarType::Type type() const { return m_type; };

		int num() const { return m_num; };

		template<typename T>
		void ApplyValue(const T& value) const;

		template<typename T>
		void ApplyValue(const T* value, int count = 1) const;

		template<typename T>
		void ApplyValue(const std::vector<T>& value) const;

		int m_handle;
	protected:
		VarType::Type m_type;
		uint16_t m_num;
	};


	class Program
	{
	public:
		Program(const Program& other) = delete;
		Program& operator=(const Program&) = delete;

		Program();
		~Program();

		bool Link(const Shader& shader);

		bool Link(const Shader& vs, const Shader& fs);

		bool Link(const Shader& vs, const Shader& gs, const Shader& fs);

		void Use() const;

		int GetAttribLocation(const char* name) const;

		int GetUniformLocation(const char* name) const;

		Uniform GetUniform(const char* name);

		Uniform GetUniform(int id) const { return m_uniforms[id]; }

		const auto& UniformMap() const { return m_uniformMap; }

	private:
		bool LinkImpl();
		void InitUniforms();

		std::vector<Uniform> m_uniforms;
		std::map<std::string, uint16_t> m_uniformMap;

		GLHandle m_program;
	};


	typedef std::shared_ptr<Program> ProgramPtr;

	ProgramPtr MakeProgram(const char* vertex_shader, const char* fragment_shader);
}

