#pragma once

#include <string>
#include<filesystem>

#include "Mesh.h"

void read_obj(const std::filesystem::path& filename, Mesh& mesh);

void save_obj(const std::filesystem::path& filename, Mesh& mesh);

void save_obj_efficient(const std::filesystem::path& filename, const Mesh& mesh);

void save_obs_binary(const std::filesystem::path& filename, Mesh& mesh);

void load_obs_binary(const std::filesystem::path& filename, Mesh& mesh);

void motion_mesh_to_binary();

void motion_binary_to_mesh();

struct MeshLog {

	static const int node_info_num { 4 * 3 };

	static const int edge_info_num { 8 };

	static const int face_info_num { 8 * 3 };

	size_t nodes_num, edges_num, faces_num;

	// Nodes Info
	std::vector<Eigen::Vector3d*> pos, pos_0, vel, acceleration;

	// Edges Info
	std::vector<double*> bend_yield_t, bend_yield_strain, bend_plastic_direction;

	std::vector<double*> bend_plastic_strain, bend_plastic_strain_hardening;

	std::vector<double*> bend_stick_t, bend_anchor_strain, bend_strain_prev;

	// Faces Info
	std::vector<double*> stretch_yield_t, stretch_yield_strain, stretch_plastic_direction;

	std::vector<double*> stretch_plastic_strain, stretch_plastic_strain_hardening;

	std::vector<double*> stretch_stick_t, stretch_anchor_strain, stretch_strain_prev;

	void connect_mesh(Mesh& mesh);

	size_t cal_element_num() const;

	void save_mesh_binary(std::filesystem::path filename) const;

	void load_mesh_binary(std::filesystem::path filename);

	MeshLog() : nodes_num{0}, edges_num{0}, faces_num{0} {};
	// Copy Constructor
	MeshLog(const MeshLog& other) = delete;
	// Copy Assignment
	MeshLog& operator=(const MeshLog& other) = delete;
	// Move Constructor
	MeshLog(MeshLog&& other) noexcept {
		this->nodes_num = other.nodes_num;
		this->edges_num = other.edges_num;
		this->faces_num = other.faces_num;

		this->pos = std::move(other.pos);
		this->pos_0 = std::move(other.pos_0);
		this->vel = std::move(other.vel);
		this->acceleration = std::move(other.acceleration);

		this->bend_yield_t = std::move(other.bend_yield_t);
		this->bend_yield_strain = std::move(other.bend_yield_strain);
		this->bend_plastic_direction = std::move(other.bend_plastic_direction);

		this->bend_plastic_strain = std::move(other.bend_plastic_strain);
		this->bend_plastic_strain_hardening = std::move(other.bend_plastic_strain_hardening);

		this->bend_stick_t = std::move(other.bend_stick_t);
		this->bend_anchor_strain = std::move(other.bend_anchor_strain);
		this->bend_strain_prev = std::move(other.bend_strain_prev);

		this->stretch_yield_t = std::move(other.stretch_yield_t);
		this->stretch_yield_strain = std::move(other.stretch_yield_strain);
		this->stretch_plastic_direction = std::move(other.stretch_plastic_direction);

		this->stretch_plastic_strain = std::move(other.stretch_plastic_strain);
		this->stretch_plastic_strain_hardening = std::move(other.stretch_plastic_strain_hardening);

		this->stretch_stick_t = std::move(other.stretch_stick_t);
		this->stretch_anchor_strain = std::move(other.stretch_anchor_strain);
		this->stretch_strain_prev = std::move(other.stretch_strain_prev);
	}
	// Move Assignment
	MeshLog& operator=(MeshLog&& other) = delete;
	// Destructor
	~MeshLog() = default;
};