#pragma once

#include "Constraint.h"
#include "AccStructure.h"

struct Min_Node;
struct Min_Edge;
struct Min_Face;

struct Proximity {

	double repulsion_thickness, repulsion_stiffness, dmin;

	std::vector<std::pair<Face const*, Face const*>>* prox_faces;

	std::vector<Mesh*> cloth_meshes, obs_meshes;

	std::vector< std::array<std::vector<Min_Face>, 2>> node_prox_current;
	std::vector< std::array<std::vector<Min_Edge>, 2>> edge_prox_current;
	std::vector< std::array<std::vector<Min_Node>, 2>> face_prox_current;

	void proximity_constraints_current(const std::vector<Mesh*>& _meshes, const std::vector<Mesh*>& _obs_meshes,
		double mu, double mu_obs, std::vector<IneqCon*>& cons_num);

	Node* get_node(int i, const std::vector<Mesh*>& meshes);

	Edge* get_edge(int i, const std::vector<Mesh*>& meshes);

	Face* get_face(int i, const std::vector<Mesh*>& meshes);

	int get_node_index(const Node* n, const std::vector<Mesh*>& meshes);

	int get_edge_index(const Edge* e, const std::vector<Mesh*>& meshes);

	int get_face_index(const Face* f, const std::vector<Mesh*>& meshes);

	int find_node_in_meshes(const Node* n, const std::vector<Mesh*>& meshes);

	int find_edge_in_meshes(const Edge* e, const std::vector<Mesh*>& meshes);

	int find_face_in_meshes(const Face* f, const std::vector<Mesh*>& meshes);

	void compute_proximities_current(const Face* face0, const Face* face1);

	bool in_wedge(double w, const Edge* edge0, const Edge* edge1);

	void add_proximity_current(const Node* node, const Face* face, int thread_idx);

	void add_proximity_current(const Edge* edge0, const Edge* edge1, int thread_idx);

	void find_proximities(const Face* face0, const Face* face1);

	void for_overlapping_faces(DeformBVHNode* node);

	void for_overlapping_faces(DeformBVHNode* node0, DeformBVHNode* node1);

	void for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, bool parallel = true);

	double area(const Node* node);

	double area(const Edge* edge);

	double area(const Face* face);

	bool is_movable(const Node* n);

	bool is_movable(const Edge* e);

	bool is_movable(const Face* f);

	IneqCon* make_constraint_num(const Node* node, const Face* face, double mu, double mu_obs);

	IneqCon* make_constraint_num(const Edge* edge0, const Edge* edge1, double mu, double mu_obs);

	std::vector<DeformBVHNode*> collect_upper_nodes(const std::vector<AccelStruct*>& accs, int num_nodes);

	// Constructor
	Proximity() : repulsion_thickness{ 0.0 }, repulsion_stiffness{0.0}, dmin { 0.0 }, prox_faces{ nullptr } {}
	
	Proximity(double& thickness, double& stiffness) : 
		repulsion_thickness{ thickness }, repulsion_stiffness{ stiffness }, dmin{ 0.0 }, prox_faces{ nullptr } {}
	
	// Copy Constructor
	Proximity(const Proximity& other) = delete;
	// Copy Assignment
	Proximity& operator=(const Proximity& other) = delete;

	// Move Constructor
	Proximity(Proximity&& other) = delete;

	// Move Assignment
	Proximity& operator=(Proximity&& other) = delete;

	// Destructor
	~Proximity() = default;
};


struct Min_Face
{
	double key;
	Face* val;

	double dist;

	void add(double key, Face* val)
	{
		if (key < this->key)
		{
			this->key = key;
			this->val = val;
		}
	}

	void add_num(double dist, Face* val) {
		if (dist < this->dist) {
			this->dist = dist;
			this->val = val;
		}
	}

	Min_Face() : key{std::numeric_limits<double>::infinity()}, val{ nullptr } {
		dist = std::numeric_limits<double>().max();
	}
	// Copy Constructor
	Min_Face(const Min_Face& other) {
		this->key = other.key;
		this->val = other.val;
		this->dist = other.dist;
	}
	// Copy Assignment
	Min_Face& operator=(const Min_Face& other) {
		if (this == &other) return *this;

		this->key = other.key;
		this->val = other.val;
		this->dist = other.dist;

		return *this;
	}

	// Move Constructor
	Min_Face(Min_Face&& other) = delete;

	// Move Assignment
	Min_Face& operator=(Min_Face&& other) = delete;

	// Destructor
	~Min_Face() = default;
};

struct Min_Edge
{
	double key;
	Edge* val;

	double dist;

	void add(double key, Edge* val)
	{
		if (key < this->key)
		{
			this->key = key;
			this->val = val;
		}
	}

	void add_num(double dist, Edge* val) {

		if (dist < this->dist) {
			this->dist = dist;
			this->val = val;
		}
	}

	Min_Edge() : key{ std::numeric_limits<double>::infinity() }, val{ nullptr } {
		dist = std::numeric_limits<double>().max();
	}

	// Copy Constructor
	Min_Edge(const Min_Edge& other) {
		this->key = other.key;
		this->val = other.val;
		this->dist = other.dist;
	}
	// Copy Assignment
	Min_Edge& operator=(const Min_Edge& other) {
		if (this == &other) return *this;

		this->key = other.key;
		this->val = other.val;
		this->dist = other.dist;

		return *this;
	}
	// Move Constructor
	Min_Edge(Min_Edge&& other) = delete;
	// Move Assignment
	Min_Edge& operator=(Min_Edge&& other) = delete;
	// Destructor
	~Min_Edge() = default;
};

struct Min_Node
{
	double key;
	Node* val;

	double dist;

	void add(double key, Node* val)
	{
		if (key < this->key)
		{
			this->key = key;
			this->val = val;
		}
	}

	void add_num(double dist, Node* val) {
		if (dist < this->dist) {
			this->dist = dist;
			this->val = val;
		}
	}

	Min_Node() : key{ std::numeric_limits<double>::infinity() }, val{ nullptr } {
		dist = std::numeric_limits<double>().max();
	}
	// Copy Constructor
	Min_Node(const Min_Node& other) {
		this->key = other.key;
		this->val = other.val;
		this->dist = other.dist;
	}
	// Copy Assignment
	Min_Node& operator=(const Min_Node& other) {
		if (this == &other) return *this;

		this->key = other.key;
		this->val = other.val;
		this->dist = other.dist;

		return *this;
	}
	// Move Constructor
	Min_Node(Min_Node&& other) = delete;
	// Move Assignment
	Min_Node& operator=(Min_Node&& other) = delete;
	// Destructor
	~Min_Node() = default;
};