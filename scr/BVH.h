#pragma once

#include "Mesh.h"

struct kDOP18 {

	double dist[18];

	void getDistances(const double p[], double& d3, double& d4, double& d5, double& d6, double& d7, double& d8) const;

	void getDistances(const double p[], double d[]) const;

	double getDistances(const double p[], int i) const;

	void getDistances(const Eigen::Vector3d& p, double& d3, double& d4, double& d5, double& d6, double& d7, double& d8) const;

	void getDistances(const Eigen::Vector3d& p, double d[]) const;

	bool overlaps(const kDOP18& b) const;

	bool overlaps(const kDOP18& b, kDOP18& ret) const;

	bool inside(const double p[]) const;

	kDOP18& operator += (const double p[]);

	kDOP18& operator += (const Eigen::Vector3d& p);

	kDOP18& operator += (const kDOP18& b);

	kDOP18 operator + (const kDOP18& v) const;

	double length(size_t i) const;

	void empty();

	double width()  const { return dist[9] - dist[0]; }
	double height() const { return dist[10] - dist[1]; }
	double depth()  const { return dist[11] - dist[2]; }
	double volume() const { return width() * height() * depth(); }
	double center(size_t i) const { return (dist[i + 9] + dist[i]) * 0.5; }

	Eigen::Vector3d center() const;

	// Default Constructor
	kDOP18();

	kDOP18(const Eigen::Vector3d v);

	kDOP18(const double v[]);

	kDOP18(const double a[], const double b[]);

	// Copy Constructor
	//kDOP18(const kDOP18& other) = default;

	kDOP18(const kDOP18& other) {

		//std::cout << "Copy Constructor" << std::endl;

		for (unsigned int i = 0; i < 18; ++i)
			this->dist[i] = other.dist[i];

	}

	// Copy Assignment
	// 
	//kDOP18& operator=(const kDOP18& other) = default;

	kDOP18& operator=(const kDOP18& other) {

		//std::cout << "Copy Assignment" << std::endl;

		if (this == &other) return *this;

		for (unsigned int i = 0; i < 18; ++i)
			this->dist[i] = other.dist[i];

		return *this;
	}


	// Move Constructor
	//kDOP18(kDOP18 && other) = default;
	
	kDOP18(kDOP18&& other) noexcept {

		//std::cout << "Move Constructor" << std::endl;

		for (unsigned int i = 0; i < 18; ++i) {
			this->dist[i] = other.dist[i];
			//other.dist[i] = 0.0;
		}
			
	}

	// Move Assignment

	//kDOP18& operator=(kDOP18&& other) = default;

	kDOP18& operator=(kDOP18&& other) noexcept {

		//std::cout << "Move Assignment" << std::endl;

		if (this == &other) return *this;

		for (unsigned int i = 0; i < 18; ++i) {
			this->dist[i] = other.dist[i];
			//other.dist[i] = 0.0;
		}
		
		return *this;
	}

	// Destructor
	virtual ~kDOP18() = default;
};

struct DeformBVHNode {

	kDOP18 box;

	Face* face;

	DeformBVHNode* parent;
	DeformBVHNode* left;
	DeformBVHNode* right;

	bool active;

	void refit(bool is_ccd = false);
	bool find(Face* face);

	inline DeformBVHNode* getLeftChild() { return left; }
	inline DeformBVHNode* getRightChild() { return right; }
	inline DeformBVHNode* getParent() { return parent; }

	inline Face* getFace() { return face; }
	inline bool isLeaf() const { return left == nullptr; }
	inline bool isRoot() const { return parent == nullptr; }

	// Constructors
	DeformBVHNode() : face{ nullptr }, parent{ nullptr }, left{ nullptr }, right{ nullptr }, active{ false } {}
	DeformBVHNode(DeformBVHNode* parent, Face* face, kDOP18* tri_boxes, Eigen::Vector3d tri_centers[]);
	DeformBVHNode(DeformBVHNode* parent, Face** lst, unsigned int lst_num, kDOP18* tri_boxes, Eigen::Vector3d tri_centers[]);

	// Copy Constructor

	//DeformBVHNode(const DeformBVHNode& other) = default;

	DeformBVHNode(const DeformBVHNode& other) {
		this->box = other.box;
		this->face = other.face;

		this->parent = other.parent;
		this->left = other.left;
		this->right = other.right;

		this->active = other.active;

	}

	// Copy Assignment
	DeformBVHNode& operator=(const DeformBVHNode& other) = delete;
	// Move Constructor
	DeformBVHNode(DeformBVHNode&& other) = delete;
	// Move Assignment
	DeformBVHNode& operator=(DeformBVHNode&& other) = delete;
	// Destructor
	//~DeformBVHNode() = default;
	
	~DeformBVHNode() {
		face = nullptr;

		parent = nullptr;
		left = nullptr;
		right = nullptr;
	}

	friend struct DeformBVHTree;
};

struct DeformBVHTree {

	Mesh* mdl;
	DeformBVHNode* root;
	Face** face_buffer;

	bool is_ccd;

	DeformBVHTree() : mdl{ nullptr }, root{ nullptr }, face_buffer{ nullptr }, is_ccd{ false } {};
	DeformBVHTree(Mesh& mdl, bool is_ccd);
	// Copy Constructor
	DeformBVHTree(const DeformBVHTree& other) = delete;
	// Copy Assignment
	DeformBVHTree& operator=(const DeformBVHTree& other) = delete;
	// Move Constructor
	DeformBVHTree(DeformBVHTree&& other) = delete;
	// Move Assignment
	DeformBVHTree& operator=(DeformBVHTree&& other) = delete;
	// Destructor
	~DeformBVHTree();

	void Construct();

	double refit();

	kDOP18 box();
	DeformBVHNode* getRoot() { return root; }

	friend struct DeformBVHNode;
};

kDOP18 vert_box(const Vert* vert, bool ccd);
kDOP18 node_box(const Node* node, bool ccd);
kDOP18 edge_box(const Edge* edge, bool ccd);
kDOP18 face_box(const Face* face, bool ccd);
bool overlap(const kDOP18& box0, const kDOP18& box1, double thickness);

kDOP18 node_box_num(const Node* node, bool ccd);