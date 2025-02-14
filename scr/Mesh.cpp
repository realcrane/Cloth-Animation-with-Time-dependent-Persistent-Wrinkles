#include "Mesh.h"

#include <numbers>

inline int tri_idx_next(int i)
{
	return i == 2 ? 0 : i + 1;
}

inline int tri_idx_prev(int i)
{
	return i == 0 ? 2 : i - 1;
}

Vert* edge_vert(const Edge* edge, int side, int i)
{
	Face* face = edge->adj_faces[side];

	if (face == nullptr)
		return nullptr;

	for (int v = 0; v < 3; ++v)
		if (face->v[v]->node == edge->nodes[i])
			return face->v[v];

	return nullptr;
}


double unwrap_angle(double theta, double theta_ref)
{
	if ((theta - theta_ref) > std::numbers::pi)
	{
		theta -= 2 * std::numbers::pi;
	}
	if ((theta - theta_ref) < -std::numbers::pi)
	{
		theta += 2 * std::numbers::pi;
	}
	return theta;
}

double Edge::dihedral_angle() const
{
	if (this->adj_faces[0] == nullptr || this->adj_faces[1] == nullptr)
		return 0.0;

	Eigen::Vector3d edge_length_ws = this->nodes[0]->x - this->nodes[1]->x;
	double edge_lenght_ws_l = edge_length_ws.norm();
	Eigen::Vector3d edge_length_ws_normal = edge_length_ws;

	if (edge_lenght_ws_l != 0.0)
		edge_length_ws_normal = edge_length_ws / edge_lenght_ws_l;

	if (edge_length_ws_normal.isZero())
		return 0.0;

	Eigen::Vector3d face0_normal = this->adj_faces[0]->face_normal();
	Eigen::Vector3d face1_normal = this->adj_faces[1]->face_normal();

	if (face0_normal.isZero() || face1_normal.isZero())
		return 0.0;

	double cosine = face0_normal.dot(face1_normal);
	double sine = edge_length_ws_normal.dot(face0_normal.cross(face1_normal));

	double theta = std::atan2(sine, cosine);

	return unwrap_angle(theta, this->reference_angle);
}

Eigen::Vector3d Face::face_normal() const
{
	const Eigen::Vector3d& x0 = this->v[0]->node->x;
	const Eigen::Vector3d& x1 = this->v[1]->node->x;
	const Eigen::Vector3d& x2 = this->v[2]->node->x;

	Eigen::Vector3d face_n = ((x1 - x0).cross(x2 - x0)).normalized();

	return face_n;
}


void Mesh::update_x0()
{
	// Update Nodes' previous position

	for (auto& n : nodes)
		n->x0 = n->x;
}

Vert* edge_opp_vert(const Edge* edge, int side)
{
	Face* face = edge->adj_faces[side];

	if (face == nullptr)
		return nullptr;

	for (int v = 0; v < 3; ++v)
		if (face->v[v]->node == edge->nodes[side])
			return face->v[tri_idx_prev(v)];

	return nullptr;
}

void Mesh::add(Vert* vert) {

	verts.push_back(vert);

	vert->node = nullptr;

	vert->index = verts.size() - 1;

	vert->adj_faces.clear();

}

void Mesh::add(Node* node) {

	nodes.push_back(node);

	for (int v = 0; v < node->verts.size(); ++v)
		node->verts[v]->node = node;

	node->index = nodes.size() - 1;

	node->adj_egdes.clear();
}

void Mesh::add(Edge* edge) {

	edges.push_back(edge);

	edge->adj_faces[0] = nullptr;

	edge->adj_faces[1] = nullptr;

	edge->index = edges.size() - 1;

	auto find_edge_0 = find(edge->nodes[0]->adj_egdes.cbegin(), edge->nodes[0]->adj_egdes.cend(), edge);

	if (find_edge_0 == edge->nodes[0]->adj_egdes.cend())
		edge->nodes[0]->adj_egdes.push_back(edge);

	auto find_edge_1 = find(edge->nodes[1]->adj_egdes.cbegin(), edge->nodes[1]->adj_egdes.cend(), edge);

	if (find_edge_1 == edge->nodes[1]->adj_egdes.cend())
		edge->nodes[1]->adj_egdes.push_back(edge);
}

void Mesh::add(Face* face) {

	faces.push_back(face);

	face->index = faces.size() - 1;

	add_edges_if_needed(face);

	for (int i = 0; i < 3; ++i)
	{
		Vert* v0 = face->v[tri_idx_next(i)];
		Vert* v1 = face->v[tri_idx_prev(i)];

		auto find_face = find(v0->adj_faces.cbegin(), v0->adj_faces.cend(), face);

		if (find_face == v0->adj_faces.cend())
			v0->adj_faces.push_back(face);

		Edge* e = get_edge(v0->node, v1->node);

		face->adj_edges[i] = e;

		int size = e->nodes[0] == v0->node ? 0 : 1;	// The order of adjacent faces are important

		e->adj_faces[size] = face;
	}
}

Edge* Mesh::get_edge(const Node* n0, const Node* n1)
{
	for (Edge* e : n0->adj_egdes)
	{
		if (e->nodes[0] == n1 || e->nodes[1] == n1)
			return e;
	}

	return nullptr;
}

void Mesh::add_edges_if_needed(const Face* face)
{
	for (int i = 0; i < 3; ++i)
	{
		Node* n0 = face->v[i]->node;
		Node* n1 = face->v[tri_idx_next(i)]->node;

		if (get_edge(n0, n1) == nullptr)
			this->add(new Edge(n0, n1));
	}
}


void Mesh::compute_ms_data()
{
	for (auto f : faces)
		compute_ms_data(f);
	for (auto e : edges)
		compute_ms_data(e);
	for (auto v : verts)
		compute_ms_data(v);
	for (auto n : nodes)
		compute_ms_data(n);
}

void Mesh::compute_ws_data()
{
	for (auto f : faces)
		compute_ws_data(f);
	for (auto e : edges)
		compute_ws_data(e);
	for (auto n : nodes)
		compute_ws_data(n);
}

void Mesh::compute_ms_data(Face* face)
{
	const Eigen::Vector2d& v0 = face->v[0]->u;
	const Eigen::Vector2d& v1 = face->v[1]->u;
	const Eigen::Vector2d& v2 = face->v[2]->u;

	face->Dm.col(0) = v1 - v0;
	face->Dm.col(1) = v2 - v0;

	face->m_a = 0.5 * face->Dm.determinant();

	if (face->m_a == 0.0)
		face->invDm = Eigen::Matrix2d::Zero();
	else
		face->invDm = face->Dm.inverse();
}


void Mesh::compute_ms_data(Edge* edge)
{

	for (int s = 0; s < 2; ++s)
		if (edge->adj_faces[s] != nullptr)
			edge->l = edge->l + (edge_vert(edge, s, 0)->u - edge_vert(edge, s, 1)->u).norm();

	if (edge->adj_faces[0] && edge->adj_faces[1])
		edge->l = edge->l / 2;

	if (!edge->adj_faces[0] || !edge->adj_faces[1])
		return;

	edge->ldaa = edge->l / (edge->adj_faces[0]->m_a + edge->adj_faces[1]->m_a);
}


void Mesh::compute_ms_data(Vert* vert)
{
	// Vert area equals to sum of the 1/3 of face's area s
	for (const Face* f : vert->adj_faces)
		vert->a = vert->a + (f->m_a / 3.0);
}

void Mesh::compute_ms_data(Node* node)
{
	// Node area equal the sum of vertices' area
	for (const Vert* v : node->verts)
		node->a = node->a + v->a;
}


void Mesh::compute_ws_data(Face* face)
{
	const Eigen::Vector3d& x0 = face->v[0]->node->x;
	const Eigen::Vector3d& x1 = face->v[1]->node->x;
	const Eigen::Vector3d& x2 = face->v[2]->node->x;

	double cross_norm = ((x1 - x0).cross(x2 - x0)).norm();

	face->w_a = 0.5 * cross_norm;
	face->n = (x1 - x0).cross(x2 - x0) / cross_norm;
}

void Mesh::compute_ws_data(Edge* edge)
{
	edge->theta = edge->dihedral_angle();
}

void Mesh::compute_ws_data(Node* node)
{
	for (const Vert* v : node->verts)
		for (const Face* f : v->adj_faces)
			node->n = node->n + f->n;

	if ( (node->n).norm() != 0.0)
		node->n = node->n / (node->n).norm();
}

void Mesh::update_norms()
{

#pragma omp parallel for
	for (int f = 0; f < faces.size(); ++f) {

		const Eigen::Vector3d& x0 = faces[f]->v[0]->node->x;
		const Eigen::Vector3d& x1 = faces[f]->v[1]->node->x;
		const Eigen::Vector3d& x2 = faces[f]->v[2]->node->x;

		Eigen::Vector3d face_n_unit = faces[f]->n;

		auto face_n = (x1 - x0).cross(x2 - x0);

		face_n_unit = face_n / face_n.norm();
	}

#pragma omp parallel for
	for (int n = 0; n < nodes.size(); ++n) {

		Eigen::Vector3d node_n = Eigen::Vector3d::Zero(3);

		for (const Vert* v : nodes[n]->verts)
			for (const Face* f : v->adj_faces) {
				const Eigen::Vector3d& face_n = f->n;
				node_n += face_n;
			}

		if (node_n.norm() != 0.0)
			node_n /= node_n.norm();

		nodes[n]->n = node_n;
	}
}

void Mesh::reset_stretch()
{
	for (Face* f: faces) {

		double a = (f->v[1]->node->x - f->v[0]->node->x).norm();
		double b = (f->v[2]->node->x - f->v[1]->node->x).norm();
		double c = (f->v[0]->node->x - f->v[2]->node->x).norm();

		double i = (
			(f->v[2]->node->x(0) - f->v[0]->node->x(0)) * (f->v[1]->node->x(0) - f->v[0]->node->x(0)) +
			(f->v[2]->node->x(1) - f->v[0]->node->x(1)) * (f->v[1]->node->x(1) - f->v[0]->node->x(1)) +
			(f->v[2]->node->x(2) - f->v[0]->node->x(2)) * (f->v[1]->node->x(2) - f->v[0]->node->x(2))) / a;
		double j = std::sqrt((f->v[2]->node->x - f->v[0]->node->x).squaredNorm() - i * i);

		Eigen::Vector2d u0 = Eigen::Vector2d::Zero();
		Eigen::Vector2d u1 = Eigen::Vector2d(a, 0.0);
		Eigen::Vector2d u2 = Eigen::Vector2d(i, j);

		f->Dm.col(0) = u1 - u0;
		f->Dm.col(1) = u2 - u0;

		f->m_a = 0.5 * f->Dm.determinant();
 
		if (f->m_a == 0.0)
			f->invDm = Eigen::Matrix2d::Zero();
		else
			f->invDm = f->Dm.inverse();
	}
}

void Mesh::mesh_info() const
{
	std::cout << "Mesh Nodes Number: " << nodes.size()
		<< "\nMesh Vertices Number: " << verts.size()
		<< "\nMesh Edges Number: " << edges.size()
		<< "\nMesh Faces Number: " << faces.size() << std::endl;
}

void delete_mesh(Mesh& mesh)
{
	for (int v = 0; v < mesh.verts.size(); v++)
		delete mesh.verts[v];
	for (int n = 0; n < mesh.nodes.size(); n++)
		delete mesh.nodes[n];
	for (int e = 0; e < mesh.edges.size(); e++)
		delete mesh.edges[e];
	for (int f = 0; f < mesh.faces.size(); f++)
		delete mesh.faces[f];

	mesh.verts.clear();
	mesh.nodes.clear();
	mesh.edges.clear();
	mesh.faces.clear();
}

void clear_mesh(Mesh& mesh)
{
	mesh.verts.clear();
	mesh.nodes.clear();
	mesh.edges.clear();
	mesh.faces.clear();
}