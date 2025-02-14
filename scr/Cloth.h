#pragma once

#include "Object.h"
#include "Proximity.h"
#include <set>

#include "Eigen/Sparse"

using sparse_tri = Eigen::Triplet<double>;
using sparse_M = Eigen::SparseMatrix<double>;

struct PlasticParameter {
	
	double k_hardening, k_hardening_0, tao, yield_ori;

	// Default Constructor
	PlasticParameter() : k_hardening{ 0.0 }, k_hardening_0{ 0.0 }, tao{ 0.0 }, yield_ori{ 0.0 } {}
	// Copy Constructor
	PlasticParameter(const PlasticParameter& other) = delete;
	// Copy Assignment
	PlasticParameter& operator=(const PlasticParameter& other) = delete;
	// Move Constructor
	PlasticParameter(PlasticParameter&& other) = delete;
	// Move Assignment
	PlasticParameter& operator=(PlasticParameter&& other) noexcept {

		if (this == &other) return *this;

		this->k_hardening = other.k_hardening;
		this->k_hardening_0 = other.k_hardening_0;
		this->tao = other.tao;
		this->yield_ori = other.yield_ori;

		other.k_hardening = 0.0;
		other.k_hardening_0 = 0.0;
		other.tao = 0.0;
		other.yield_ori = 0.0;

		return *this;
	}

	// Destructor
	virtual ~PlasticParameter() = default;
};

struct FrictionParameter
{
	double thres_0, thres_inf, tao;

	double k, C1, C2, k_0, k_c, vs;

	// Default Constructor
	FrictionParameter() : thres_0{ 0.0 }, thres_inf{ 0.0 }, tao{0.0},
		k{0.0}, C1 {0.0}, C2{ 0.0 }, k_0{ 0.0 }, k_c{ 0.0 }, vs{ 0.0 } {}
	// Copy Constructor
	FrictionParameter(const FrictionParameter& other) = delete;
	// Copy Assignment
	FrictionParameter& operator=(const FrictionParameter& other) = delete;
	// Move Constructor
	FrictionParameter(FrictionParameter&& other) = delete;
	// Move Assignment
	FrictionParameter& operator=(FrictionParameter&& other) noexcept {

		if (this == &other) return *this;

		this->thres_0 = other.thres_0;
		this->thres_inf = other.thres_inf;
		this->tao = other.tao;

		this->C1 = other.C1;
		this->C2 = other.C2;

		this->k = other.k;
		this->k_0 = other.k_0;
		this->k_c = other.k_c;
		this->vs = other.vs;

		other.thres_0 = 0.0;
		other.thres_0 = 0.0;
		other.tao = 0.0;

		other.C1 = 0.0;
		other.C2 = 0.0;

		other.k = 0.0;
		other.k_0 = 0.0;
		other.k_c = 0.0;
		other.vs = 0.0;

		return *this;
	}

	// Destructor
	virtual ~FrictionParameter() = default;
};

struct Cloth : Object
{	
	bool is_bending_plastic, is_debug_bending_plastic;
	bool is_bending_friction, is_debug_bending_friction;

	bool is_tensile_plastic, is_debug_tensile_plastic;
	bool is_tensile_friction, is_debug_tensile_friction;

	bool is_handle_on;

	bool is_stable;

	std::set<unsigned int> stable_steps;
	
	std::array<PlasticParameter, 3> tensile_plastic_parameters;
	
	PlasticParameter bend_plastic_parameters;

	std::array<FrictionParameter, 3> tensile_friction_parameters;

	FrictionParameter bend_friction_parameters;

	//std::vector<double> force_tracker;
	
	std::vector<Handle> handles;
	
	void object_type() override { std::cout << "Cloth" << std::endl; }

	void set_handles();

	void update_handles(const int current_step);

	void renew_anchors();

	void stable_nodes();

	void initialize_mesh_parameters();

	void zero_stretch_strains();

	void zero_bend_strains();

	void cal_mass(std::vector<sparse_tri>& mass_triplet, const Eigen::Vector3d& gravity);	// Calculate Mass Matrix

	void cal_external_force(Eigen::VectorXd& Cloth_Force, const Eigen::Vector3d& gravity, const double& dt);	// Calculate External Forces

	void cal_stretch(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, const double& dt);		// Calculate Stretching Force/Jacobian

	void cal_bend(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, const double& dt);		// Calculate Bending Force/Jacobian

	void cal_constraint(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, std::vector<IneqCon*>& cons, const double& dt);

	void cal_handle_constrains(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, const int current_step, const double& stiffness, const double& dt);

	void time_elapse(double duration);

	Cloth() : is_bending_plastic{ false }, is_debug_bending_plastic{false}, 
		is_bending_friction{ false }, is_debug_bending_friction{ false },
		is_tensile_plastic{ false }, is_debug_tensile_plastic{ false },
		is_tensile_friction{ false }, is_debug_tensile_friction{ false },
		is_handle_on{ false }, is_stable{false} {
		
		std::cout << "Create Cloth" << std::endl;

		for (unsigned int i = 0; i < 3; ++i) {
			tensile_plastic_parameters.at(i) = PlasticParameter();
			tensile_friction_parameters.at(i) = FrictionParameter();
		}
			
		bend_plastic_parameters = PlasticParameter();
		bend_friction_parameters = FrictionParameter();
	}

	// Copy Constructor
	Cloth(const Cloth& other) = delete;
	// Copy Assignment
	Cloth& operator=(const Cloth& other) = delete;
	
	// Move Constructor
	Cloth(Cloth&& other) noexcept {

		std::cout << "Move Constructor in Cloth class is called" << std::endl;

		this->mesh = std::move(other.mesh);

		this->is_bending_plastic = other.is_bending_plastic;

		this->is_debug_bending_plastic = other.is_debug_bending_plastic;

		this->is_bending_friction = other.is_bending_friction;

		this->is_debug_bending_friction = other.is_debug_bending_friction;

		this->is_tensile_plastic = other.is_tensile_plastic;

		this->is_debug_tensile_plastic = other.is_debug_tensile_plastic;

		this->is_tensile_friction = other.is_tensile_friction;

		this->is_debug_tensile_friction = other.is_debug_tensile_friction;

		this->is_handle_on = other.is_handle_on;

		this->is_stable = other.is_stable;

		this->stable_steps = std::move(other.stable_steps);

		this->material = std::move(other.material);

		this->tensile_plastic_parameters = std::move(other.tensile_plastic_parameters);

		this->tensile_friction_parameters = std::move(other.tensile_friction_parameters);

		this->bend_plastic_parameters = std::move(other.bend_plastic_parameters);

		this->bend_friction_parameters = std::move(other.bend_friction_parameters);

		this->handles = std::move(other.handles);
	}

	// Move Assignment
	Cloth& operator=(Cloth&& other) = delete;

	// Destructor
	~Cloth() = default;

};
