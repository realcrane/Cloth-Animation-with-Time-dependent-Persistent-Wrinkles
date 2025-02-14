#pragma once
#include <string>
#include <vector>

#include "glm/glm.hpp"


struct VertexGL {
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
};

struct TextureGL {
	unsigned int id;
	std::string type;
};

class MeshGL {
public:
	// mesh data
	std::vector<VertexGL> vertices;
	std::vector<unsigned int> indices;
	std::vector<TextureGL> textures;


	// Defualt Constructor
	MeshGL() = default;
	// Constructors
	MeshGL(std::vector<VertexGL> vertices, std::vector<unsigned int> indices, std::vector<TextureGL> textures);
	// Copy Constructor
	MeshGL(const MeshGL& other) = delete;
	// Copy Assignment
	MeshGL& operator=(const MeshGL& other) = delete;
	// Move Constructor
	MeshGL(MeshGL&& other) = delete;
	// Move Assignment
	MeshGL& operator=(MeshGL&& other) = delete;
	// Destructor
	~MeshGL() = default;

	void Draw(Shader& shader);

private:
	unsigned int VAO, VBO, EBO;

	void setupMesh();
};