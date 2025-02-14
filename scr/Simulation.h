#pragma once

#include "Cloth.h"
#include "Environment.h"
#include "Obstacle.h"
#include <filesystem>

enum class SolverType {
	EigenCG,
	CPUCG,
	CPUPCG,
	CUDACG,
	GPUCG,
	GPUPCG,
	Test
};

struct Simulation
{	
	double dt;

	int step_num, current_step;

	bool is_collision, is_proximity;

	double repulsion_thickness, repulsion_stiffness, collision_thickness;

	double cloth_mu, cloth_obs_mu;

	bool is_profile_time, is_profile_solver, is_print_step;

	bool is_elapse;

	size_t elapse_start, elapse_end;

	double elapse_duration;

	bool is_save_mesh, is_save_binary, is_load_binary;

	int save_mesh_per_steps;

	SolverType solver_type;

	Environment env;

	std::filesystem::path mesh_save_path, binary_save_path, binary_load_path;
	
	std::vector<Cloth> clothes;

	std::vector<Obstacle> obstacles;
	std::vector<std::pair<bool, std::filesystem::path>> obstacle_is_save_and_pathes;
	
	void physics();

	// Default Constructor
	Simulation() : dt{ 0.0 }, step_num{ 0 }, current_step {0}, collision_thickness { 5e-3 },
		is_collision{ true }, is_proximity{ true },
		repulsion_thickness{ 1e-2 }, repulsion_stiffness{ 1e3 },
		cloth_mu{ 0.5 }, cloth_obs_mu { 0.2 },
		is_profile_time{ false }, is_profile_solver{ false }, is_print_step {false}, is_elapse{false},
		elapse_start{ std::numeric_limits<size_t>::max() }, elapse_end{ std::numeric_limits<size_t>::max() },
		elapse_duration{ 0.0 }, is_save_mesh{ false }, is_save_binary{ false }, is_load_binary { false },
		save_mesh_per_steps{ 1 }, solver_type { SolverType::EigenCG } {
		std::cout << "Default Simulation Constructor" << std::endl;
	}

	Simulation(std::filesystem::path config_path);

	// Copy Constructor
	Simulation(const Simulation& other) = delete;
	// Copy Assignment
	Simulation& operator=(const Simulation& other) = delete;
	// Move Constructor
	Simulation(Simulation&& other) = delete;
	// Move Assignment
	Simulation& operator=(Simulation&& other) = delete;

	// Destructor
	~Simulation() = default;
};
