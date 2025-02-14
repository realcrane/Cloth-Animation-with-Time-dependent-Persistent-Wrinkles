#pragma once

#include "Eigen/Dense"
#include <numbers>

enum class MotionType {
	Fixed,
	Translation,
	Rotation
};

struct Motion {

	virtual void Move(Eigen::Vector3d& x) = 0;	// pure virtual class

	Motion() {};
	// Copy Constructor
	Motion(const Motion& other) = delete;
	// Copy Assignment
	Motion& operator=(const Motion& other) = delete;
	// Move Constructor
	Motion(Motion&& other) = delete;
	// Move Assignment
	Motion& operator=(Motion&& other) = delete;
	// Destructor
	virtual ~Motion() = default;
};


struct Fixed : Motion {
	
	void Move(Eigen::Vector3d& x) override {
		//std::cout << "Fixed Handle" << std::endl;
	}

	Fixed() {};
	// Copy Constructor
	Fixed(const Fixed& other) = delete;
	// Copy Assignment
	Fixed& operator=(const Fixed& other) = delete;
	// Move Constructor
	Fixed(Fixed&& other) = delete;
	// Move Assignment
	Fixed& operator=(Fixed&& other) = default;
	// Destructor
	~Fixed() = default;
};

struct Translation : Motion
{
	Eigen::Vector3d TranslationVec;

	void Move(Eigen::Vector3d& x) override {
		x += TranslationVec;
	}

	Translation() {
		TranslationVec = Eigen::Vector3d::Zero();
	}

	Translation(double trans_x, double trans_y, double trans_z) {
		TranslationVec(0) = trans_x;
		TranslationVec(1) = trans_y;
		TranslationVec(2) = trans_z;
	}

	// Copy Constructor
	Translation(const Translation& other) = delete;
	// Copy Assignment
	Translation& operator=(const Translation& other) = delete;
	// Move Constructor
	Translation(Translation&& other) = delete;
	// Move Assignment
	Translation& operator=(Translation&& other) noexcept {

		if (this == &other) return *this;

		this->TranslationVec = other.TranslationVec;

		return *this;
	}

	// Destructor
	~Translation() = default;
};

struct Rotation : Motion
{
	Eigen::Vector3d Center;
	Eigen::Matrix3d RotMat, World_to_Local_Mat;

	Eigen::Vector3d operator()(Eigen::Vector3d x, double theta_x, double theta_y, double theta_z, 
		double theta_world_to_local_x, double theta_world_to_local_y, double theta_world_to_local_z) {

		Eigen::Matrix3d Rot_loc_x, Rot_loc_y, Rot_loc_z;

		Eigen::Matrix3d Rot_world_to_local_x, Rot_world_to_local_y, Rot_world_to_local_z;

		Rot_loc_x << 1.0, 0.0, 0.0,
			0.0, cos(theta_x), -sin(theta_x),
			0.0, sin(theta_x), cos(theta_x);

		Rot_loc_y << cos(theta_y), 0.0, sin(theta_y),
			0.0, 1.0, 0.0,
			-sin(theta_y), 0, cos(theta_y);

		Rot_loc_z << cos(theta_z), -sin(theta_z), 0.0,
			sin(theta_z), cos(theta_z), 0.0,
			0.0, 0.0, 1.0;

		Rot_world_to_local_x << 1.0, 0.0, 0.0,
			0.0, cos(theta_world_to_local_x), -sin(theta_world_to_local_x),
			0.0, sin(theta_world_to_local_x), cos(theta_x);

		Rot_world_to_local_y << cos(theta_world_to_local_y), 0.0, sin(theta_world_to_local_y),
			0.0, 1.0, 0.0,
			-sin(theta_world_to_local_y), 0, cos(theta_world_to_local_y);

		Rot_world_to_local_z << cos(theta_world_to_local_z), -sin(theta_world_to_local_z), 0.0,
			sin(theta_world_to_local_z), cos(theta_world_to_local_z), 0.0,
			0.0, 0.0, 1.0;

		Eigen::Matrix3d Rot = Rot_loc_z * Rot_loc_z * Rot_loc_x;
		Eigen::Matrix3d Rot_world_to_local = Rot_world_to_local_z * Rot_world_to_local_y * Rot_world_to_local_x;

		return ((Rot_world_to_local.transpose() * Rot * Rot_world_to_local * (x - Center)) + Center);
	}

	void Move(Eigen::Vector3d& x) override {
		//x = RotMat * (x - Center) + Center;
		x = ( ( World_to_Local_Mat.transpose() * (RotMat * (World_to_Local_Mat * (x - Center))) ) + Center);
	}

	Rotation() {
		Center = Eigen::Vector3d::Zero();
		RotMat = Eigen::Matrix3d::Zero();
		World_to_Local_Mat = Eigen::Matrix3d::Zero();
	}

	Rotation(Eigen::Vector3d center, double theta_x, double theta_y, double theta_z,
		double theta_world_to_local_x, double theta_world_to_local_y, double theta_world_to_local_z) {
		
		Center = center;

		Eigen::Matrix3d Rot_loc_x, Rot_loc_y, Rot_loc_z;

		Eigen::Matrix3d Rot_world_to_local_x, Rot_world_to_local_y, Rot_world_to_local_z;

		Rot_loc_x << 1.0, 0.0, 0.0,
			0.0, cos(theta_x), -sin(theta_x),
			0.0, sin(theta_x), cos(theta_x);

		Rot_loc_y << cos(theta_y), 0.0, sin(theta_y),
			0.0, 1.0, 0.0,
			-sin(theta_y), 0, cos(theta_y);

		Rot_loc_z << cos(theta_z), -sin(theta_z), 0.0,
			sin(theta_z), cos(theta_z), 0.0,
			0.0, 0.0, 1.0;

		Rot_world_to_local_x << 1.0, 0.0, 0.0,
			0.0, cos(theta_world_to_local_x), -sin(theta_world_to_local_x),
			0.0, sin(theta_world_to_local_x), cos(theta_x);

		Rot_world_to_local_y << cos(theta_world_to_local_y), 0.0, sin(theta_world_to_local_y),
			0.0, 1.0, 0.0,
			-sin(theta_world_to_local_y), 0.0, cos(theta_world_to_local_y);

		Rot_world_to_local_z << cos(theta_world_to_local_z), -sin(theta_world_to_local_z), 0.0,
			sin(theta_world_to_local_z), cos(theta_world_to_local_z), 0.0,
			0.0, 0.0, 1.0;

		RotMat = Rot_loc_z * Rot_loc_z * Rot_loc_x;
		World_to_Local_Mat = Rot_world_to_local_z * Rot_world_to_local_y * Rot_world_to_local_x;
	}

	// Copy Constructor
	Rotation(const Rotation& other) = delete;
	// Copy Assignment
	Rotation& operator=(const Rotation& other) = delete;
	// Move Constructor
	Rotation(Rotation&& other) = delete;
	// Move Assignment
	Rotation& operator=(Rotation&& other) noexcept  {
		
		if (this == &other) return *this;

		this->Center = other.Center;
		this->RotMat = other.RotMat;
		this->World_to_Local_Mat = other.World_to_Local_Mat;

		return *this;
	}
	
	// Destructor
	~Rotation() = default;
};