#pragma once
#include "BVH.h"

struct AccelStruct
{
	DeformBVHTree tree;
	DeformBVHNode* root;

	std::vector<DeformBVHNode*> leaves;

	// Default constructor
	AccelStruct() : root{nullptr} {
		//std::cout << "Create AccelStruct" << std::endl;
	}
	AccelStruct(const Mesh& mesh, bool is_ccd);
	// Copy Constructor
	AccelStruct(const AccelStruct& other) = delete;
	// Copy Assignment
	AccelStruct& operator=(const AccelStruct& other) = delete;
	// Move Constructor
	AccelStruct(AccelStruct&& other) = delete;
	// Move Assignment
	AccelStruct& operator=(AccelStruct&& other) = delete;
	// Destructor
	~AccelStruct() {
		//std::cout << "Delete AccelStruct" << std::endl;
	}
};

std::vector<AccelStruct*> create_accel_structs(const std::vector<Mesh*>& meshes, bool ccd);

void update_accel_struct(AccelStruct& acc);

void destroy_accel_structs(std::vector<AccelStruct*>& accs);
