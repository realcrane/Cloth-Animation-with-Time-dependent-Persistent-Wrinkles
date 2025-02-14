#pragma once

#include "Impact.h"

struct Collision {

	std::vector<Mesh*> cloth_meshes, obs_meshes;

	std::vector<Impact>* impacts;

	std::vector<std::pair<Face const*, Face const*>>* faceimpacts;

	int* cnt;

	unsigned int nthreads;

	void build_node_lookup(const std::vector<Mesh*>& cloth_meshes);

	void collision_response(std::vector<Mesh*>& meshes, const std::vector<Mesh*>& obs_meshes, const double& thickness, const double& time_step, const bool is_profile_time = false, const unsigned int max_iter = 100);

	void update_active(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, const std::vector<ImpactZone*>& zones);

	std::vector<Impact> find_impacts(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, const double& thickness, const bool is_profile_time = false);

	void for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, double thickness, bool parallel=true);

	void for_overlapping_faces(DeformBVHNode* node0, DeformBVHNode* node1, const double& thickness);

	void for_overlapping_faces(DeformBVHNode* node, const double& thickness);

	void find_face_impacts(const Face* face0, const Face* face1);

	void compute_face_impacts(const Face* face0, const Face* face1, const double& thickness);

	bool vf_collision_test(const Vert* vert, const Face* face, Impact& impact, const double& thickness);	// Test Vertex Face Collision

	bool ee_collision_test(const Edge* edge0, const Edge* edge1, Impact& impact, const double& thickness);	// Test Edge Edge Collision

	bool collision_test(Impact::Type type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact, const double& thickness);

	std::vector<Impact> independent_impacts(const std::vector<Impact>& impacts);

	void add_impacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones);

	void merge_zones(ImpactZone* zone0, ImpactZone* zone1, std::vector<ImpactZone*>& zones);

	ImpactZone* find_or_create_zone(const Node* node, std::vector<ImpactZone*>& zones);

	void exclude(const ImpactZone* z, std::vector<ImpactZone*>& zs);

	void remove_zone_from_zones(int i, std::vector<ImpactZone*>& zs);

	int find_zone_in_zones(const ImpactZone* z, std::vector<ImpactZone*> zs);

	void apply_inelastic_projection(ImpactZone* zone, const double& thickness, bool verbose = false);

	Collision() : impacts{ nullptr }, faceimpacts{nullptr}, cnt { nullptr }, nthreads{ 0 } {
		std::cout << "Create Collision" << std::endl;
	}

	// Copy Constructor
	Collision(const Collision& other) = delete;

	// Copy Assignment
	Collision& operator=(const Collision& other) = delete;

	// Move Constructor
	Collision(Collision&& other) = delete;

	// Move Assignment
	Collision& operator=(Collision&& other) = delete;

	// Destructor
	~Collision() = default;
};