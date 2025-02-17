#pragma once

#include "Mesh.h"
#include "Motion.h"

struct Handle {

	Node* node;

	bool is_motion;

	std::vector<std::unique_ptr<Motion>> motions;

	std::vector<std::pair<int, int>> start_end_step;

	Eigen::Vector3d anchor_pos;

	void update(size_t motion_idx) {
		if (is_motion) motions[motion_idx]->Move(anchor_pos);
	}

	void renew_anchor() { anchor_pos = node->x; }

	Handle() : node{ nullptr }, is_motion{ false } {
		anchor_pos << 0.0, 0.0, 0.0;
	}

	Handle(Node* node) : is_motion{ false } {
		this->node = node;
		anchor_pos = node->x;
	}

	Handle(Motion* motion) : node{ nullptr }, is_motion{ false } {
		Eigen::Vector3d x = Eigen::Vector3d::Zero();
		motion->Move(x);
	}

	Handle(Node* node, Motion* motion): is_motion{ false } {
		this->node = node;
		anchor_pos = node->x;
	}

	// Copy Constructor
	Handle(const Handle& other) = delete;
	// Copy Assignment
	Handle& operator=(const Handle& other) = delete;
	// Move Constructor
	Handle(Handle&& other) noexcept {
		this->node = other.node;
		this->anchor_pos = other.anchor_pos;
		this->is_motion = other.is_motion;
		this->motions = std::move(other.motions);
		this->start_end_step = std::move(other.start_end_step);

		other.node = nullptr;
		other.anchor_pos = Eigen::Vector3d::Zero();
	}
	// Move Assignment
	Handle& operator=(Handle&& other) noexcept {

		if (this == &other) return *this;

		this->node = other.node;
		this->anchor_pos = other.anchor_pos;
		this->is_motion = other.is_motion;
		this->motions = std::move(other.motions);
		this->start_end_step = std::move(other.start_end_step);

		other.node = nullptr;
		other.anchor_pos = Eigen::Vector3d::Zero();

		return *this;
	}

	// Destructor
	virtual ~Handle() {

		this->node = nullptr;
		this->anchor_pos = Eigen::Vector3d::Zero();
				
	}

};