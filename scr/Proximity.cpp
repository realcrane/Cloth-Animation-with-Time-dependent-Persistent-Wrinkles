#include <random>

#include "Proximity.h"

static std::vector<Min_Face> node_prox[2];
static std::vector<Min_Edge> edge_prox[2];
static std::vector<Min_Node> face_prox[2];

inline int NEXT(int i)
{
	return (i < 2) ? (i + 1) : (i - 2);
}

inline int PREV(int i)
{
	return (i > 0) ? (i - 1) : (i + 2);
}

void Proximity::proximity_constraints_current(const std::vector<Mesh*>& cloth_meshes, const std::vector<Mesh*>& obs_meshes, double mu, double mu_obs, std::vector<IneqCon*>& cons_num)
{
	int nthreads = omp_get_max_threads();

	if (!prox_faces)
		prox_faces = new std::vector<std::pair<Face const*, Face const*>>[nthreads];

	this->cloth_meshes = cloth_meshes;
	this->obs_meshes = obs_meshes;

	dmin = 2 * repulsion_thickness;

	std::vector<AccelStruct*> accs = create_accel_structs(this->cloth_meshes, false);
	std::vector<AccelStruct*> obs_accs = create_accel_structs(this->obs_meshes, false);

	int num_nodes = 0;
	int num_edges = 0;
	int num_faces = 0;

	for (int m = 0; m < this->cloth_meshes.size(); ++m) {
		num_nodes = num_nodes + static_cast<int>(this->cloth_meshes[m]->nodes.size());
		num_edges = num_edges + static_cast<int>(this->cloth_meshes[m]->edges.size());
		num_faces = num_faces + static_cast<int>(this->cloth_meshes[m]->faces.size());
	}

	for (int t = 0; t < nthreads; ++t) {
		std::array<std::vector<Min_Face>, 2> node_prox = std::array<std::vector<Min_Face>, 2>();
		std::array<std::vector<Min_Edge>, 2> edge_prox = std::array<std::vector<Min_Edge>, 2>();
		std::array<std::vector<Min_Node>, 2> face_prox = std::array<std::vector<Min_Node>, 2>();
		for (int s = 0; s < 2; ++s) {
			node_prox[s].assign(num_nodes, Min_Face());
			edge_prox[s].assign(num_edges, Min_Edge());
			face_prox[s].assign(num_faces, Min_Node());
		}
		node_prox_current.emplace_back(node_prox);
		edge_prox_current.emplace_back(edge_prox);
		face_prox_current.emplace_back(face_prox);
	}

	for_overlapping_faces(accs, obs_accs, true);

	std::vector<std::pair<Face const*, Face const*> > tot_faces;
	for (int t = 0; t < nthreads; ++t)
		tot_faces.insert(tot_faces.end(), prox_faces[t].begin(), prox_faces[t].end());

	// random_shuffle deprecated
	std::random_device rng;
	std::mt19937 urng(rng());
	std::shuffle(tot_faces.begin(), tot_faces.end(), urng);

#pragma omp parallel for
	for (int i = 0; i < tot_faces.size(); ++i)
		compute_proximities_current(tot_faces[i].first, tot_faces[i].second);

	for (int i = 0; i < 2; ++i) {
		::node_prox[i].assign(num_nodes, Min_Face());
		::edge_prox[i].assign(num_edges, Min_Edge());
		::face_prox[i].assign(num_faces, Min_Node());
	}

	double dist = std::numeric_limits<double>().max();

	for (int s = 0; s < 2; ++s) {
		for (int n = 0; n < num_nodes; ++n) {
			dist = std::numeric_limits<double>().max();
			Face* dist_face = nullptr;
			for (int t = 0; t < nthreads; ++t) {
				if (node_prox_current[t][s][n].dist < dist) {
					dist = node_prox_current[t][s][n].dist;
					dist_face = node_prox_current[t][s][n].val;
				}
				::node_prox[s][n].val = dist_face;
				::node_prox[s][n].dist = dist;
			}
		}

		for (int e = 0; e < num_edges; ++e) {
			dist = std::numeric_limits<double>().max();
			Edge* dist_edge = nullptr;
			for (int t = 0; t < nthreads; ++t) {
				if (edge_prox_current[t][s][e].dist < dist) {
					dist = edge_prox_current[t][s][e].dist;
					dist_edge = edge_prox_current[t][s][e].val;
				}
				::edge_prox[s][e].val = dist_edge;
				::edge_prox[s][e].dist = dist;
			}
		}

		for (int f = 0; f < num_faces; ++f) {
			dist = std::numeric_limits<double>().max();
			Node* dist_node = nullptr;
			for (int t = 0; t < nthreads; ++t) {
				if (face_prox_current[t][s][f].dist < dist) {
					dist = face_prox_current[t][s][f].dist;
					dist_node = face_prox_current[t][s][f].val;
				}
				::face_prox[s][f].val = dist_node;
				::face_prox[s][f].dist = dist;
			}
		}
	}

	for (int n = 0; n < num_nodes; n++)
		for (int i = 0; i < 2; i++)
		{
			Min_Face& m = ::node_prox[i][n];
			if (m.dist < dmin)
				cons_num.push_back(make_constraint_num(get_node(n, cloth_meshes), m.val, mu, mu_obs));
		}

	for (int e = 0; e < num_edges; e++)
		for (int i = 0; i < 2; i++)
		{
			Min_Edge& m = ::edge_prox[i][e];
			if (m.dist < dmin)
				cons_num.push_back(make_constraint_num(get_edge(e, cloth_meshes), m.val, mu, mu_obs));
		}

	for (int f = 0; f < num_faces; f++)
		for (int i = 0; i < 2; i++)
		{
			Min_Node& m = ::face_prox[i][f];
			if (m.dist < dmin)
				cons_num.push_back(make_constraint_num(m.val, get_face(f, cloth_meshes), mu, mu_obs));
		}

	for (int t = 0; t < nthreads; ++t) {
		for (int s = 0; s < 2; ++s) {
			face_prox[s].clear();
			face_prox[s].clear();
			face_prox[s].clear();
		}
	}

	destroy_accel_structs(accs);
	destroy_accel_structs(obs_accs);

	delete[] prox_faces;
}

std::vector<DeformBVHNode*> Proximity::collect_upper_nodes(const std::vector<AccelStruct*>& accs, int num_nodes)
{
	std::vector<DeformBVHNode*> nodes;
	for (int a = 0; a < accs.size(); a++)
		if (accs[a]->root)
			nodes.push_back(accs[a]->root);
	while (nodes.size() < num_nodes) {
		std::vector<DeformBVHNode*> children;
		for (int n = 0; n < nodes.size(); n++)
			if (nodes[n]->isLeaf())
				children.push_back(nodes[n]);
			else {
				children.push_back(nodes[n]->left);
				children.push_back(nodes[n]->right);
			}
		if (children.size() == nodes.size())
			break;
		nodes = children;
	}
	return nodes;
}

void Proximity::find_proximities(const Face* face0, const Face* face1)
{
	int t = omp_get_thread_num();
	prox_faces[t].push_back(std::make_pair(face0, face1));
}

void Proximity::for_overlapping_faces(DeformBVHNode* node)
{
	if (node->isLeaf() || !node->active)
		return;
	for_overlapping_faces(node->getLeftChild());
	for_overlapping_faces(node->getRightChild());
	for_overlapping_faces(node->getLeftChild(), node->getRightChild());
}

void Proximity::for_overlapping_faces(DeformBVHNode* node0, DeformBVHNode* node1)
{
	if (!node0->active && !node1->active)
		return;
	if (!overlap(node0->box, node1->box, dmin))
		return;	
	if (node0->isLeaf() && node1->isLeaf()) {
		Face* face0 = node0->getFace(),
			* face1 = node1->getFace();
		find_proximities(face0, face1);
	}
	else if (node0->isLeaf()) {
		for_overlapping_faces(node0, node1->getLeftChild());
		for_overlapping_faces(node0, node1->getRightChild());
	}
	else {
		for_overlapping_faces(node0->getLeftChild(), node1);
		for_overlapping_faces(node0->getRightChild(), node1);
	}
}

void Proximity::for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, bool parallel)
{
	int nnodes = static_cast<int>(ceil(sqrt(2 * omp_get_max_threads())));
	std::vector<DeformBVHNode*> nodes = collect_upper_nodes(accs, nnodes);
	int nthreads = omp_get_max_threads();
	omp_set_num_threads(parallel ? omp_get_max_threads() : 1);
#pragma omp parallel for
	for (int n = 0; n < nodes.size(); n++) {
		for_overlapping_faces(nodes[n]);
		for (int m = 0; m < n; m++)
			for_overlapping_faces(nodes[n], nodes[m]);
		for (int o = 0; o < obs_accs.size(); o++)
			if (obs_accs[o]->root)
				for_overlapping_faces(nodes[n], obs_accs[o]->root);
	}

	// Obstacle-Obstacle Repulse
//	nodes = collect_upper_nodes(obs_accs, nnodes);
//#pragma omp parallel for
//	for (int n = 0; n < nodes.size(); n++) {
//		for_overlapping_faces(nodes[n], thickness);
//		for (int m = 0; m < n; m++)
//			for_overlapping_faces(nodes[n], nodes[m], thickness);
//	}

	omp_set_num_threads(nthreads);
}

void Proximity::compute_proximities_current(const Face* face0, const Face* face1)
{
	int t = omp_get_thread_num();

	kDOP18 nb[6], eb[6], fb[2];

	for (int v = 0; v < 3; ++v) {
		nb[v] = node_box_num(face0->v[v]->node, false);
		nb[v + 3] = node_box_num(face1->v[v]->node, false);
	}

	for (int v = 0; v < 3; ++v) {
		eb[v] = nb[NEXT(v)] + nb[PREV(v)];//edge_box(face0->adje[v], true);//
		eb[v + 3] = nb[NEXT(v) + 3] + nb[PREV(v) + 3];//edge_box(face1->adje[v], true);//
	}

	fb[0] = nb[0] + nb[1] + nb[2];
	fb[1] = nb[3] + nb[4] + nb[5];

	for (int v = 0; v < 3; v++) {
		if (!overlap(nb[v], fb[1], dmin))
			continue;
		add_proximity_current(face0->v[v]->node, face1, t);
	}

	for (int v = 0; v < 3; v++) {
		if (!overlap(nb[v + 3], fb[0], dmin))
			continue;
		add_proximity_current(face1->v[v]->node, face0, t);
	}

	for (int e0 = 0; e0 < 3; e0++)
		for (int e1 = 0; e1 < 3; e1++) {
			if (!overlap(eb[e0], eb[e1 + 3], dmin))
				continue;
			add_proximity_current(face0->adj_edges[e0], face1->adj_edges[e1], t);
		}

}

void Proximity::add_proximity_current(const Node* node, const Face* face, int thread_idx)
{
	if (node == face->v[0]->node ||
		node == face->v[1]->node ||
		node == face->v[2]->node)
		return;
		
	const Eigen::Vector3d x_pos = node->x;
	const Eigen::Vector3d f_x0_pos = face->v[0]->node->x;
	const Eigen::Vector3d f_x1_pos = face->v[1]->node->x;
	const Eigen::Vector3d f_x2_pos = face->v[2]->node->x;

	auto n = (f_x1_pos - f_x0_pos).cross(f_x2_pos - f_x0_pos);

	if (n.dot(n) < 1e-12) {
		return;
	}

	n /= n.norm();

	auto h = (-(f_x0_pos - x_pos)).dot(n);

	if (h > dmin) {
		return;
	}

	auto b0 = (f_x1_pos - x_pos).dot((f_x2_pos - x_pos).cross(n));
	auto b1 = (f_x2_pos - x_pos).dot((f_x0_pos - x_pos).cross(n));
	auto b2 = (f_x0_pos - x_pos).dot((f_x1_pos - x_pos).cross(n));

	auto sum = 1.0 / (b0 + b1 + b2);

	auto w1 = -b0 * sum;
	auto w2 = -b1 * sum;
	auto w3 = -b2 * sum;

	bool inside = (std::min(std::min(-w1, -w2), -w3) >= -1e-12);
	if (!inside)
		return;

	const Eigen::Vector3d node_n = node->n;
	const Eigen::Vector3d face_n = face->n;

	if (is_movable(node)) {
		int side = (n.dot(node_n) >= 0) ? 0 : 1;
		node_prox_current[thread_idx][side][get_node_index(node, cloth_meshes)].add_num(h, (Face*)face);
	}

	if (is_movable(face)) {
		int side = (-n.dot(face_n) >= 0) ? 0 : 1;
		face_prox_current[thread_idx][side][get_face_index(face, cloth_meshes)].add_num(h, (Node*)node);
	}
}

void Proximity::add_proximity_current(const Edge* edge0, const Edge* edge1, int thread_idx)
{
	if (edge0->nodes[0] == edge1->nodes[0] ||
		edge0->nodes[0] == edge1->nodes[1] ||
		edge0->nodes[1] == edge1->nodes[0] ||
		edge0->nodes[1] == edge1->nodes[1])
		return;
		
	const Eigen::Vector3d e0_x0_pos = edge0->nodes[0]->x;
	const Eigen::Vector3d e0_x1_pos = edge0->nodes[1]->x;
	const Eigen::Vector3d e1_x0_pos = edge1->nodes[0]->x;
	const Eigen::Vector3d e1_x1_pos = edge1->nodes[1]->x;

	auto n = (e0_x1_pos - e0_x0_pos).cross(e1_x1_pos - e1_x0_pos);

	if (n.dot(n) < 1e-12) {
		return;
	}

	n /= n.norm();

	auto h = (-(e1_x0_pos - e0_x0_pos)).dot(n);

	if (abs(h) > dmin) {
		return;
	}

	auto a0 = (e1_x1_pos - e0_x1_pos).dot((e1_x0_pos - e0_x1_pos).cross(n));
	auto a1 = (e1_x0_pos - e0_x0_pos).dot((e1_x1_pos - e0_x0_pos).cross(n));

	auto b0 = (e1_x1_pos - e0_x1_pos).dot((e1_x1_pos - e0_x0_pos).cross(n));
	auto b1 = (e1_x0_pos - e0_x0_pos).dot((e1_x0_pos - e0_x1_pos).cross(n));

	auto sum_a = 1.0 / (a0 + a1);
	auto sum_b = 1.0 / (b0 + b1);

	auto w0 = a0 * sum_a;
	auto w1 = a1 * sum_a;
	auto w2 = -b0 * sum_b;
	auto w3 = -b1 * sum_b;

	bool inside = std::min(std::min(w0, w1), std::min(-w2, -w3)) >= -1e-12 && in_wedge(w1, edge0, edge1) && in_wedge(-w3, edge1, edge0);

	if (!inside) {
		return;
	}

	if (is_movable(edge0)) {
		const Eigen::Vector3d e0_x0_n = edge0->nodes[0]->n;
		const Eigen::Vector3d e0_x1_n = edge0->nodes[1]->n;
		auto edge0n = e0_x0_n + e0_x1_n;
		int side = (n.dot(edge0n) >= 0) ? 0 : 1;
		edge_prox_current[thread_idx][side][get_edge_index(edge0, cloth_meshes)].add_num(h, (Edge*)edge1);
	}

	if (is_movable(edge1)) {
		const Eigen::Vector3d e1_x0_n = edge0->nodes[0]->n;
		const Eigen::Vector3d e1_x1_n = edge0->nodes[1]->n;
		auto edge1n = e1_x0_n + e1_x1_n;
		int side = (-n.dot(edge1n) >= 0) ? 0 : 1;
		edge_prox_current[thread_idx][side][get_edge_index(edge1, cloth_meshes)].add_num(h, (Edge*)edge0);
	}
}

bool Proximity::in_wedge(double w, const Edge* edge0, const Edge* edge1)
{
	const Eigen::Vector3d e0_x0_pos = edge0->nodes[0]->x;
	const Eigen::Vector3d e0_x1_pos = edge0->nodes[1]->x;
	Eigen::Vector3d x = (1.0 - w) * e0_x0_pos + w * e0_x1_pos;
	bool in{ true };
	for (int s = 0; s < 2; ++s) {
		const Face* face = edge1->adj_faces[s];
		if (!face)
			continue;
		const Eigen::Vector3d e1_x0_pos = edge1->nodes[s]->x;
		const Eigen::Vector3d e1_x1_pos = edge1->nodes[1 - s]->x;
		const Eigen::Vector3d face_n = face->n;
		Eigen::Vector3d e = e1_x1_pos - e1_x0_pos;
		Eigen::Vector3d r = x - e1_x0_pos;
		in &= ((e.dot(face_n.cross(r))) > 0);
	}
	return in;
}

bool Proximity::is_movable(const Node* n)
{
	return find_node_in_meshes(n, cloth_meshes) != -1;
}

bool Proximity::is_movable(const Edge* e)
{
	return find_edge_in_meshes(e, cloth_meshes) != -1;
}

bool Proximity::is_movable(const Face* f)
{
	return find_face_in_meshes(f, cloth_meshes) != -1;
}

Node* Proximity::get_node(int i, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Node*>& nodes = meshes[m]->nodes;
		if (i < nodes.size())
			return nodes[i];
		else
			i -= nodes.size();
	}
	return nullptr;
}

Edge* Proximity::get_edge(int i, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Edge*>& edges = meshes[m]->edges;
		if (i < edges.size())
			return edges[i];
		else
			i -= edges.size();
	}
	return nullptr;
}

Face* Proximity::get_face(int i, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Face*>& faces = meshes[m]->faces;
		if (i < faces.size())
			return faces[i];
		else
			i -= faces.size();
	}
	return nullptr;
}

int Proximity::find_node_in_meshes(const Node* n, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Node*>& nodes = meshes[m]->nodes;

		if (n->index < nodes.size() && n == nodes[n->index])
		{
			return m;
		}
	}

	return -1;
}

int Proximity::find_edge_in_meshes(const Edge* e, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Edge*>& edges = meshes[m]->edges;

		if (e->index < edges.size() && e == edges[e->index])
		{
			return m;
		}
	}

	return -1;
}

int Proximity::find_face_in_meshes(const Face* f, const std::vector<Mesh*>& meshes)
{	
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Face*>& faces = meshes[m]->faces;


		if (f->index < faces.size() && f == faces[f->index])
		{
			
			return m;
		}
	}

	return -1;
}

int Proximity::get_node_index(const Node* n, const std::vector<Mesh*>& meshes)
{
	int i = 0;
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Node*>& nodes = meshes[m]->nodes;
		if (n->index < nodes.size() && n == nodes[n->index])
			return i + n->index;
		else
			i += nodes.size();
	}

	return -1;
}

int Proximity::get_edge_index(const Edge* e, const std::vector<Mesh*>& meshes)
{
	int i = 0;
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Edge*>& edges = meshes[m]->edges;
		if (e->index < edges.size() && e == edges[e->index])
			return i + e->index;
		else
			i += edges.size();
	}
	return -1;
}

int Proximity::get_face_index(const Face* f, const std::vector<Mesh*>& meshes)
{
	int i = 0;
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Face*>& faces = meshes[m]->faces;
		if (f->index < faces.size() && f == faces[f->index])
			return i + f->index;
		else
			i += faces.size();
	}
	return -1;
}

double Proximity::area(const Node* node)
{
	return node->a;
}

double Proximity::area(const Edge* edge)
{
	double a = 0.0;

	if (edge->adj_faces[0])
		a = a + area(edge->adj_faces[0]) / 3.0;
	if (edge->adj_faces[1])
		a = a + area(edge->adj_faces[1]) / 3.0;

	return a;
}

double Proximity::area(const Face* face)
{
	if (is_movable(face))
		return face->m_a;
	else
		return face->w_a;
}

IneqCon* Proximity::make_constraint_num(const Node* node, const Face* face, double mu, double mu_obs) {

	IneqCon* con = new IneqCon;
	con->nodes[0] = (Node*)node;
	con->nodes[1] = (Node*)face->v[0]->node;
	con->nodes[2] = (Node*)face->v[1]->node;
	con->nodes[3] = (Node*)face->v[2]->node;
	for (int n = 0; n < 4; n++)
		con->free[n] = is_movable(con->nodes[n]);
	double a = std::min(area(node), area(face));
	con->stiff = repulsion_stiffness * a;
	con->repulsion_thickness = repulsion_thickness;

	const Eigen::Vector3d x_pos = node->x;
	const Eigen::Vector3d f_x0_pos = face->v[0]->node->x;
	const Eigen::Vector3d f_x1_pos = face->v[1]->node->x;
	const Eigen::Vector3d f_x2_pos = face->v[2]->node->x;

	auto n = (f_x1_pos - f_x0_pos).cross(f_x2_pos - f_x0_pos);

	n /= n.norm();

	auto h = (-(f_x0_pos - x_pos)).dot(n);

	auto b0 = (f_x1_pos - x_pos).dot((f_x2_pos - x_pos).cross(n));
	auto b1 = (f_x2_pos - x_pos).dot((f_x0_pos - x_pos).cross(n));
	auto b2 = (f_x0_pos - x_pos).dot((f_x1_pos - x_pos).cross(n));

	auto sum = 1.0 / (b0 + b1 + b2);

	auto w1 = -b0 * sum;
	auto w2 = -b1 * sum;
	auto w3 = -b2 * sum;

	con->w[0] = 1.0;
	con->w[1] = w1;
	con->w[2] = w2;
	con->w[3] = w3;

	con->n = n;

	if (h < 0.0) {
		con->n = -con->n;
	}

	con->mu = ((!is_movable(node) || !is_movable(face)) ? mu_obs : mu);

	return con;
}

IneqCon* Proximity::make_constraint_num(const Edge* edge0, const Edge* edge1, double mu, double mu_obs) {
	IneqCon* con = new IneqCon;
	con->nodes[0] = (Node*)edge0->nodes[0];
	con->nodes[1] = (Node*)edge0->nodes[1];
	con->nodes[2] = (Node*)edge1->nodes[0];
	con->nodes[3] = (Node*)edge1->nodes[1];
	for (int n = 0; n < 4; n++)
		con->free[n] = is_movable(con->nodes[n]);
	double a = std::min(area(edge0), area(edge1));
	con->stiff = repulsion_stiffness * a;
	con->repulsion_thickness = repulsion_thickness;

	const Eigen::Vector3d e0_x0_pos = edge0->nodes[0]->x;
	const Eigen::Vector3d e0_x1_pos = edge0->nodes[1]->x;
	const Eigen::Vector3d e1_x0_pos = edge1->nodes[0]->x;
	const Eigen::Vector3d e1_x1_pos = edge1->nodes[1]->x;

	auto n = (e0_x1_pos - e0_x0_pos).cross(e1_x1_pos - e1_x0_pos);

	n /= n.norm();

	auto h = (-(e1_x0_pos - e0_x0_pos)).dot(n);

	auto a0 = (e1_x1_pos - e0_x1_pos).dot((e1_x0_pos - e0_x1_pos).cross(n));
	auto a1 = (e1_x0_pos - e0_x0_pos).dot((e1_x1_pos - e0_x0_pos).cross(n));

	auto b0 = (e1_x1_pos - e0_x1_pos).dot((e1_x1_pos - e0_x0_pos).cross(n));
	auto b1 = (e1_x0_pos - e0_x0_pos).dot((e1_x0_pos - e0_x1_pos).cross(n));

	auto sum_a = 1.0 / (a0 + a1);
	auto sum_b = 1.0 / (b0 + b1);

	auto w0 = a0 * sum_a;
	auto w1 = a1 * sum_a;
	auto w2 = -b0 * sum_b;
	auto w3 = -b1 * sum_b;

	con->w[0] = w0;
	con->w[1] = w1;
	con->w[2] = w2;
	con->w[3] = w3;

	con->n = n;

	if (h < 0.0) {
		con->n = -con->n;
	}

	con->mu = ((!is_movable(edge0) || !is_movable(edge1)) ? mu_obs : mu);

	return con;
}