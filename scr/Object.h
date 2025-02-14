#pragma once

#include "Handle.h"

struct Material {
	
	double density;
	std::array<double, 4> stretching;
	double bending;
	double damping;

	Material() : density{ 0.0 }, stretching{ 0.0, 0.0, 0.0, 0.0 }, bending{ 0.0 }, damping{ 0.0 } {}

	// Copy Constructor
	Material(const Material& other) = delete;
	// Copy Assignment
	Material& operator=(const Material& other) = delete;
	// Move Constructor
	Material(Material&& other) = delete;
	// Move Assignment
	Material& operator=(Material&& other) noexcept{
		
		if (this == &other) return *this;
		
		this->density = other.density;
		this->stretching = std::move(other.stretching);
		this->bending = other.bending;
		this->damping = other.damping;

		other.density = 0.0;
		other.stretching.fill(0.0);
		other.bending = 0.0;
		other.damping = 0.0;

		return *this;
	}

	// Destructor
	virtual ~Material() = default;
};

struct Object {

	Mesh mesh;

	Material material;

	virtual void object_type() = 0;	// pure virtual class

	// Constructor
	Object() :mesh{ Mesh() }, material{Material()} {}
	// Copy Constructor
	Object(const Object& other) = delete;
	// Copy Assignment
	Object& operator=(const Object& other) = delete;
	// Move Constructor
	Object(Object&& other) = delete;	
	// Move Assignment
	Object& operator=(Object&& other) = delete;
	
	// Destructor
	virtual ~Object() = default;
};
