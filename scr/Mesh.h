#pragma once

#include <array>
#include <vector>

#include "Eigen/Dense"
#include <iostream>

struct FricPlasState {

	double yield_t, yield_strain, plastic_direction;

	double plastic_strain, plastic_strain_hardening;

	double stick_t, anchor_strain, strain_prev;

	// Constructors
	FricPlasState() : yield_t{0.0}, yield_strain{}, plastic_direction{},
		plastic_strain{ 0.0 }, plastic_strain_hardening{ 0.0 },
		stick_t{ 0.0 }, anchor_strain{ 0.0 }, strain_prev{ 0.0 } { }
	// Copy Constructor
	FricPlasState(const FricPlasState& other) = delete;
	// Copy Assignment
	FricPlasState& operator=(const FricPlasState& other) = delete;
	// Move Constructor
	FricPlasState(FricPlasState&& other) = delete;
	// Move Assignment
	FricPlasState& operator=(FricPlasState&& other) = delete;
	// Destructor
	~FricPlasState() = default;
};

struct Node;
struct Edge;
struct Face;

struct Vert
{
	int label;

	size_t index;	// position in mesh.verts

	Eigen::Vector2d u; // material space

	double a; // area

	double m; // mass

	Node* node;

	std::vector<Face*> adj_faces;	// adjacent faces

	// Constructors
	Vert() :
		label{ 0 }, index{ 0 }, a{ 0.0 }, m{ 0.0 }
	{
		u << 0.0, 0.0;

		node = nullptr;
	}

	Vert(const Eigen::Vector3d& x, int label = 0) :
		label{label}, index{ 0 }, a{ 0.0 }, m{ 0.0 }
	{
		u = x.head(2);

		node = nullptr;
	}

	// Copy Constructor
	Vert(const Vert& other) = delete;
	// Copy Assignment
	Vert& operator=(const Vert& other) = delete;
	// Move Constructor
	Vert(Vert&& other) = delete;
	// Move Assignment
	Vert& operator=(Vert&& other) = delete;
	// Destructor
	~Vert() = default;
};

struct Node
{
	int label;

	size_t index; // position in mesh.nodes

	double m;	// Node's mass

	double a;	// Node's area

	Eigen::Vector3d x, x0;	// Node's current position and previous position

	Eigen::Vector3d v;	// Node's velocity

	Eigen::Vector3d n;	// Node's normal

	Eigen::Vector3d acceleration;

	std::vector<Vert*> verts;	// A node can be shared by multiple vertices

	std::vector<Edge*> adj_egdes;	// adjacent edges

	Node() :
		label{ 0 }, index{ 0 }, m{ 0.0 }, a{ 0.0 }
	{
		x << 0.0, 0.0, 0.0;
		x0 << 0.0, 0.0, 0.0;

		v << 0.0, 0.0, 0.0;

		n << 0.0, 0.0, 0.0;

		acceleration << 0.0, 0.0, 0.0;
	}

	Node(const Eigen::Vector3d& x, const Eigen::Vector3d& v, int label = 0) :
		label{ label }, index{ 0 }, m{ 0.0 }, a{ 0.0 }
	{
		this->x = x;
		this->x0 = x;

		this->v = v;

		this->n = Eigen::Vector3d::Zero();
	}

	// Copy Constructor
	Node(const Node& other) = delete;
	// Copy Assignment
	Node& operator=(const Node& other) = delete;
	// Move Constructor
	Node(Node&& other) = delete;
	// Move Assignment
	Node& operator=(Node&& other) = delete;
	// Destructor
	~Node() = default;
};


struct Edge
{
	int label;

	size_t index;	// position in mesh.edges

	double theta;	// actual dihedral angle

	double reference_angle;	// just to get sign of dihedral_angle() right

	double l, ldaa;	// Terms for compute bending Force

	double rest_theta;

	FricPlasState friction_plastic_state;

	std::array<Node*, 2> nodes;	// two end nodes

	std::array<Face*, 2> adj_faces;	// adjacent Faces

	double dihedral_angle() const;

	Edge() :
		label{ 0 }, index{ 0 }, theta{ 0.0 }, reference_angle{ 0.0 },
		l{ 0.0 }, ldaa{ 0.0 }, rest_theta{ 0.0 }
	{
		friction_plastic_state.yield_strain = 0.0;
		friction_plastic_state.plastic_direction = 0.0;
		friction_plastic_state.yield_t = 0.0;
		friction_plastic_state.plastic_strain = 0.0;
		friction_plastic_state.plastic_strain_hardening = 0.0;

		friction_plastic_state.stick_t = 0.0;
		friction_plastic_state.anchor_strain = 0.0;
		friction_plastic_state.strain_prev = 0.0;
		
		nodes[0] = nullptr;
		nodes[1] = nullptr;

		adj_faces[0] = nullptr;
		adj_faces[1] = nullptr;
	}

	Edge(Node* node0, Node* node1, int label = 0):
		label{ label }, index{ 0 }, theta{0.0}, reference_angle{ 0.0 },
		l{ 0.0 }, ldaa{ 0.0 }, rest_theta{ 0.0 }
	{
		nodes[0] = node0;
		nodes[1] = node1;

		adj_faces[0] = nullptr;
		adj_faces[1] = nullptr;
	}

	// Copy Constructor
	Edge(const Edge& other) = delete;
	// Copy Assignment
	Edge& operator=(const Edge& other) = delete;
	// Move Constructor
	Edge(Edge&& other) = delete;
	// Move Assignment
	Edge& operator=(Edge&& other) = delete;
	// Destructor
	~Edge() = default;
};


struct Face
{
	int label;

	size_t index; // position in mesh.faces

	double m; // Face Mass
	double m_a; // Face Area in Material Space
	double w_a; // Face Area in World Space

	std::array<double, 3> rest_strain;

	std::array<FricPlasState, 3> friction_plastic_states;

	Eigen::Matrix2d Dm, invDm;

	Eigen::Vector3d n; // Face normal

	std::array<Vert*, 3> v; // Vertices

	std::array<Edge*, 3> adj_edges;	// adjacent edges

	Eigen::Vector3d face_normal() const;

	Face() :
		label{ 0 }, index{ 0 }, m{ 0.0 }, m_a{ 0.0 }, w_a{ 0.0 }	
	{
		for (unsigned int i = 0; i < 3; ++i) {
			rest_strain[i] = 0.0;
			
			friction_plastic_states[i].yield_strain = 0.0;
			friction_plastic_states[i].plastic_direction = 0.0;
			friction_plastic_states[i].yield_t = 0.0;
			friction_plastic_states[i].plastic_strain = 0.0;
			friction_plastic_states[i].plastic_strain_hardening = 0.0;

			friction_plastic_states[i].stick_t = 0.0;
			friction_plastic_states[i].anchor_strain = 0.0;
			friction_plastic_states[i].strain_prev = 0.0;
		}
		
		Dm = Eigen::Matrix2d::Zero();
		invDm = Eigen::Matrix2d::Zero();

		n = Eigen::Vector3d::Zero();

		v[0] = nullptr;
		v[1] = nullptr;
		v[2] = nullptr;

		adj_edges[0] = nullptr;
		adj_edges[1] = nullptr;
		adj_edges[2] = nullptr;
	}

	Face(Vert* v1, Vert* v2, Vert* v3, int _label = 0):
		label{ 0 }, index{ 0 }, m{ 0.0 }, m_a{ 0.0 }, w_a{ 0.0 }
	{
		for (unsigned int i = 0; i < 3; ++i) {
			rest_strain[i] = 0.0;
			
			friction_plastic_states[i].yield_strain = 0.0;
			friction_plastic_states[i].plastic_direction = 0.0;
			friction_plastic_states[i].yield_t = 0.0;
			friction_plastic_states[i].plastic_strain = 0.0;
			friction_plastic_states[i].plastic_strain_hardening = 0.0;

			friction_plastic_states[i].stick_t = 0.0;
			friction_plastic_states[i].anchor_strain = 0.0;
			friction_plastic_states[i].strain_prev = 0.0;
		}

		Dm = Eigen::Matrix2d::Zero();
		invDm = Eigen::Matrix2d::Zero();
		
		v[0] = v1;
		v[1] = v2;
		v[2] = v3;

		adj_edges[0] = nullptr;
		adj_edges[1] = nullptr;
		adj_edges[2] = nullptr;
	}

	// Copy Constructor
	Face(const Face& other) = delete;
	// Copy Assignment
	Face& operator=(const Face& other) = delete;
	// Move Constructor
	Face(Face&& other) = delete;
	// Move Assignment
	Face& operator=(Face&& other) = delete;
	// Destructor
	~Face() = default;
};

Vert* edge_opp_vert(const Edge* edge, int side);

struct Mesh
{
	bool is_cloth;

	std::vector<Vert*> verts;
	std::vector<Node*> nodes;
	std::vector<Face*> faces;
	std::vector<Edge*> edges;

	void add(Vert* vert);
	void add(Node* node);
	void add(Edge* edge);
	void add(Face* face);

	Edge* get_edge(const Node* n0, const Node* n1);
	void add_edges_if_needed(const Face* face);

	void update_x0();

	void compute_ms_data();
	void compute_ms_data(Face* face);
	void compute_ms_data(Edge* edge);
	void compute_ms_data(Vert* vert);
	void compute_ms_data(Node* node);

	void compute_ws_data();
	void compute_ws_data(Face* face);
	void compute_ws_data(Edge* edge);
	void compute_ws_data(Node* node);

	void update_norms();
	void reset_stretch();

	void mesh_info () const;

	//Default Constructor
	Mesh() : is_cloth{ true } {}

	// Copy Constructor
	Mesh(const Mesh& other)
	{
		this->is_cloth = other.is_cloth;

		this->verts = other.verts;
		this->nodes = other.nodes;
		this->faces = other.faces;
		this->edges = other.edges;
	}

	// Copy Assignment
	Mesh& operator=(const Mesh& other)
	{
		if (this == &other) return *this;
		
		this->is_cloth = other.is_cloth;

		this->verts = other.verts;
		this->nodes = other.nodes;
		this->faces = other.faces;
		this->edges = other.edges;

		return *this;
	}

	// Move Constructor
	Mesh(Mesh&& other) = delete;
	
	// Move Assignment
	Mesh& operator=(Mesh&& other) noexcept {

		std::cout << "Move Assignment in Mesh class is called" << std::endl;
		
		if (this == &other) return *this;

		for (int v = 0; v < verts.size(); v++)
			delete verts[v];
		for (int n = 0; n < nodes.size(); n++)
			delete nodes[n];
		for (int e = 0; e < edges.size(); e++)
			delete edges[e];
		for (int f = 0; f < faces.size(); f++)
			delete faces[f];

		verts.clear();
		nodes.clear();
		edges.clear();
		faces.clear();

		this->is_cloth = other.is_cloth;

		for (size_t v = 0; v < other.verts.size(); v++)
			this->verts.push_back(other.verts[v]);
		for (size_t n = 0; n < other.nodes.size(); n++)
			this->nodes.push_back(other.nodes[n]);
		for (size_t e = 0; e < other.edges.size(); e++)
			this->edges.push_back(other.edges[e]);
		for (size_t f = 0; f < other.faces.size(); f++)
			this->faces.push_back(other.faces[f]);

		for (size_t v = 0; v < other.verts.size(); v++)
			other.verts[v] = nullptr;
		for (size_t n = 0; n < other.nodes.size(); n++)
			other.nodes[n] = nullptr;
		for (size_t e = 0; e < other.edges.size(); e++)
			other.edges[e] = nullptr;
		for (size_t f = 0; f < other.faces.size(); f++)
			other.faces[f] = nullptr;

		other.verts.clear();
		other.nodes.clear();
		other.edges.clear();
		other.faces.clear();

		return *this;
	}
	
	//Destructor
	~Mesh() {

		std::cout << "Mesh Destructor Called" << std::endl;

		for (int v = 0; v < verts.size(); v++)
			delete verts[v];
		for (int n = 0; n < nodes.size(); n++)
			delete nodes[n];
		for (int e = 0; e < edges.size(); e++)
			delete edges[e];
		for (int f = 0; f < faces.size(); f++)
			delete faces[f];

		verts.clear();
		nodes.clear();
		edges.clear();
		faces.clear();
	}
};

void delete_mesh(Mesh& mesh);

void clear_mesh(Mesh& mesh);