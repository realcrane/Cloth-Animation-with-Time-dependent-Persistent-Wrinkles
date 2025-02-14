#pragma once
#include "Object.h"
#include <filesystem>

struct Obstacle: Object {

	bool is_motion, is_deform, is_collision_active;

	bool is_load_init_binary;

	std::vector<std::unique_ptr<Motion>> motions;

	std::vector<std::pair<int, int>> motion_start_end_step, collision_start_end_step;

	std::filesystem::path deform_binary_path;

	std::filesystem::path prev_binary_path, init_binary_path;

	void object_type() override { std::cout << "Obstacle" << std::endl; }

	void cal_mass();	// Calculate Mesh Face/Node Mass

	void load_init_binary(const double dt);

	void excute_motion(const int& current_step, const double& dt);

	void excute_deformation(const int& current_step, const int& offset, const double& dt);

	void update_state(const int current_step);

	Obstacle() : is_motion{ false }, is_deform{ false }, is_collision_active{ true }, is_load_init_binary{ false } {
		std::cout << "Create Obstacle" << std::endl;
	}
	// Copy Constructor
	Obstacle(const Obstacle& other) = delete;
	// Copy Assignment
	Obstacle& operator=(const Obstacle& other) = delete;
	// Move Constructor
	Obstacle(Obstacle&& other) noexcept {

		std::cout << "Move Constructor in Cloth class is called" << std::endl;

		this->mesh = std::move(other.mesh);

		this->material = std::move(other.material);

		this->is_motion = other.is_motion;

		this->is_deform = other.is_deform;

		this->is_collision_active = other.is_collision_active;

		this->is_load_init_binary = other.is_load_init_binary;

		this->motions = std::move(other.motions);

		this->motion_start_end_step = std::move(other.motion_start_end_step);

		this->deform_binary_path = std::move(other.deform_binary_path);

		this->prev_binary_path = std::move(other.prev_binary_path);

		this->init_binary_path = std::move(other.init_binary_path);
	}
	// Move Assignment
	Obstacle& operator=(Obstacle&& other) = delete;

	// Destructor
	~Obstacle() = default;
};