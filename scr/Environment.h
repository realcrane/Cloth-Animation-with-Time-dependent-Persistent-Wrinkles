#pragma once

#include "Eigen/Dense"

struct Environment
{
	double handle_stiffness;
	
	Eigen::Vector3d gravity;

	Environment(): handle_stiffness(1e6) {
		gravity = Eigen::Vector3d(0.0, 0.0, -9.8);
	}

	// Copy Constructor
	Environment(const Environment& other) = delete;
	// Copy Assignment
	Environment& operator=(const Environment& other) = delete;

	// Move Constructor
	Environment(Environment&& other) = delete;

	// Move Assignment
	Environment& operator=(Environment&& other) noexcept {
		if (this == &other) return *this;

		this->handle_stiffness = other.handle_stiffness;
		this->gravity = other.gravity;

		return *this;
	}

	// Destructor
	~Environment() = default;
};