#include "BVH.h"

inline double MAX(double a, double b) {
	return a > b ? a : b;
}

inline double MIN(double a, double b) {
	return a < b ? a : b;
}

inline Eigen::Vector3d middle_xyz(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3)
{
	Eigen::Vector3d ans = Eigen::Vector3d::Zero();

	for (int i = 0; i < 3; ++i)
		ans[i] = 0.5 * (MIN(MIN(p1[i], p2[i]), p3[i]) + MAX(MAX(p1[i], p2[i]), p3[i]));

	return ans;
}

kDOP18 node_box(const Node* node, bool ccd)
{
	kDOP18 box(node->x);

	if (ccd) 
		box += node->x0;

	return box;
}


kDOP18 vert_box(const Vert* vert, bool ccd)
{
	return node_box(vert->node, ccd);
}

kDOP18 edge_box(const Edge* edge, bool ccd)
{
	kDOP18 box = node_box(edge->nodes[0], ccd);

	box += node_box(edge->nodes[1], ccd);

	return box;
}

kDOP18 face_box(const Face* face, bool ccd)
{
	kDOP18 box = vert_box(face->v[0], ccd);

	for (int v = 1; v < 3; v++)
		box += vert_box(face->v[v], ccd);

	return box;
}


kDOP18 node_box_num(const Node* node, bool ccd)
{
	kDOP18 box(node->x);

	if (ccd) {
		box += node->x0;
	}

	return box;
}


kDOP18 dilate(const kDOP18& box, double d)
{
	static double sqrt2 = std::sqrt(2.0);

	kDOP18 dbox = box;

	for (int i = 0; i < 3; i++)
	{
		dbox.dist[i] -= d;
		dbox.dist[i + 9] += d;
	}

	for (int i = 0; i < 6; i++)
	{
		dbox.dist[3 + i] -= sqrt2 * d;
		dbox.dist[3 + i + 9] += sqrt2 * d;
	}

	return dbox;
}

bool overlap(const kDOP18& box0, const kDOP18& box1, double thickness)
{
	return box0.overlaps(dilate(box1, thickness));
}

struct aap {

	unsigned int _xyz;
	double _p;

	bool inside(Eigen::Vector3d mid) const {
		return mid[_xyz] > _p;
	}

	// Default Constructor
	aap() = default;

	aap(const kDOP18& total) {
		Eigen::Vector3d center = total.center();
		unsigned int xyz { 2 };

		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		}
		else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

		_xyz = xyz;
		_p = center[xyz];
	}

	// Copy Constructor
	aap(const aap& other) = delete;
	// Copy Assignment
	aap& operator=(const aap& other) = delete;
	// Move Constructor
	aap(aap&& other) = delete;
	// Move Assignment
	aap& operator=(aap&& other) = delete;
	// Destructor
	~aap() = default;
};

kDOP18::kDOP18() {
	empty();
}

kDOP18::kDOP18(const Eigen::Vector3d v)
{
	dist[0] = dist[9] = v[0];
	dist[1] = dist[10] = v[1];
	dist[2] = dist[11] = v[2];

	double d3{}, d4{}, d5{}, d6{}, d7{}, d8{};

	getDistances(v, d3, d4, d5, d6, d7, d8);

	dist[3] = dist[12] = d3;
	dist[4] = dist[13] = d4;
	dist[5] = dist[14] = d5;
	dist[6] = dist[15] = d6;
	dist[7] = dist[16] = d7;
	dist[8] = dist[17] = d8;
}

kDOP18::kDOP18(const double v[]) {

	dist[0] = dist[9] = v[0];
	dist[1] = dist[10] = v[1];
	dist[2] = dist[11] = v[2];

	double d3{}, d4{}, d5{}, d6{}, d7{}, d8{};

	getDistances(v, d3, d4, d5, d6, d7, d8);

	dist[3] = dist[12] = d3;
	dist[4] = dist[13] = d4;
	dist[5] = dist[14] = d5;
	dist[6] = dist[15] = d6;
	dist[7] = dist[16] = d7;
	dist[8] = dist[17] = d8;
}

kDOP18::kDOP18(const double a[], const double b[]) {
	dist[0] = MIN(a[0], b[0]);
	dist[9] = MAX(a[0], b[0]);
	dist[1] = MIN(a[1], b[1]);
	dist[10] = MAX(a[1], b[1]);
	dist[2] = MIN(a[2], b[2]);
	dist[11] = MAX(a[2], b[2]);

	double ad3{}, ad4{}, ad5{}, ad6{}, ad7{}, ad8{};
	getDistances(a, ad3, ad4, ad5, ad6, ad7, ad8);
	double bd3{}, bd4{}, bd5{}, bd6{}, bd7{}, bd8{};
	getDistances(b, bd3, bd4, bd5, bd6, bd7, bd8);
	dist[3] = MIN(ad3, bd3);
	dist[12] = MAX(ad3, bd3);
	dist[4] = MIN(ad4, bd4);
	dist[13] = MAX(ad4, bd4);
	dist[5] = MIN(ad5, bd5);
	dist[14] = MAX(ad5, bd5);
	dist[6] = MIN(ad6, bd6);
	dist[15] = MAX(ad6, bd6);
	dist[7] = MIN(ad7, bd7);
	dist[16] = MAX(ad7, bd7);
	dist[8] = MIN(ad8, bd8);
	dist[17] = MAX(ad8, bd8);
}

kDOP18& kDOP18::operator += (const double p[])
{
	dist[0] = MIN(p[0], dist[0]);
	dist[9] = MAX(p[0], dist[9]);
	dist[1] = MIN(p[1], dist[1]);
	dist[10] = MAX(p[1], dist[10]);
	dist[2] = MIN(p[2], dist[2]);
	dist[11] = MAX(p[2], dist[11]);

	double d3{}, d4{}, d5{}, d6{}, d7{}, d8{};
	getDistances(p, d3, d4, d5, d6, d7, d8);
	dist[3] = MIN(d3, dist[3]);
	dist[12] = MAX(d3, dist[12]);
	dist[4] = MIN(d4, dist[4]);
	dist[13] = MAX(d4, dist[13]);
	dist[5] = MIN(d5, dist[5]);
	dist[14] = MAX(d5, dist[14]);
	dist[6] = MIN(d6, dist[6]);
	dist[15] = MAX(d6, dist[15]);
	dist[7] = MIN(d7, dist[7]);
	dist[16] = MAX(d7, dist[16]);
	dist[8] = MIN(d8, dist[8]);
	dist[17] = MAX(d8, dist[17]);

	return *this;
}

kDOP18& kDOP18::operator+=(const Eigen::Vector3d& p)
{
	dist[0] = MIN(p[0], dist[0]);
	dist[9] = MAX(p[0], dist[9]);
	dist[1] = MIN(p[1], dist[1]);
	dist[10] = MAX(p[1], dist[10]);
	dist[2] = MIN(p[2], dist[2]);
	dist[11] = MAX(p[2], dist[11]);

	double d3{}, d4{}, d5{}, d6{}, d7{}, d8{};
	getDistances(p, d3, d4, d5, d6, d7, d8);
	dist[3] = MIN(d3, dist[3]);
	dist[12] = MAX(d3, dist[12]);
	dist[4] = MIN(d4, dist[4]);
	dist[13] = MAX(d4, dist[13]);
	dist[5] = MIN(d5, dist[5]);
	dist[14] = MAX(d5, dist[14]);
	dist[6] = MIN(d6, dist[6]);
	dist[15] = MAX(d6, dist[15]);
	dist[7] = MIN(d7, dist[7]);
	dist[16] = MAX(d7, dist[16]);
	dist[8] = MIN(d8, dist[8]);
	dist[17] = MAX(d8, dist[17]);

	return *this;
}


kDOP18& kDOP18::operator += (const kDOP18& b)
{
	dist[0] = MIN(b.dist[0], dist[0]);
	dist[9] = MAX(b.dist[9], dist[9]);
	dist[1] = MIN(b.dist[1], dist[1]);
	dist[10] = MAX(b.dist[10], dist[10]);
	dist[2] = MIN(b.dist[2], dist[2]);
	dist[11] = MAX(b.dist[11], dist[11]);
	dist[3] = MIN(b.dist[3], dist[3]);
	dist[12] = MAX(b.dist[12], dist[12]);
	dist[4] = MIN(b.dist[4], dist[4]);
	dist[13] = MAX(b.dist[13], dist[13]);
	dist[5] = MIN(b.dist[5], dist[5]);
	dist[14] = MAX(b.dist[14], dist[14]);
	dist[6] = MIN(b.dist[6], dist[6]);
	dist[15] = MAX(b.dist[15], dist[15]);
	dist[7] = MIN(b.dist[7], dist[7]);
	dist[16] = MAX(b.dist[16], dist[16]);
	dist[8] = MIN(b.dist[8], dist[8]);
	dist[17] = MAX(b.dist[17], dist[17]);
	return *this;
}


kDOP18 kDOP18::operator + (const kDOP18& v) const
{
	kDOP18 rt(*this);
	return rt += v;
}


double kDOP18::length(size_t i) const {
	return dist[i + 9] - dist[i];
}

void kDOP18::getDistances(const double p[], double& d3, double& d4, double& d5, double& d6, double& d7, double& d8) const {
	d3 = p[0] + p[1];
	d4 = p[0] + p[2];
	d5 = p[1] + p[2];
	d6 = p[0] - p[1];
	d7 = p[0] - p[2];
	d8 = p[1] - p[2];
}


void kDOP18::getDistances(const double p[], double d[]) const {
	d[0] = p[0] + p[1];
	d[1] = p[0] + p[2];
	d[2] = p[1] + p[2];
	d[3] = p[0] - p[1];
	d[4] = p[0] - p[2];
	d[5] = p[1] - p[2];
}


double kDOP18::getDistances(const double p[], int i) const
{
	if (i == 0) return p[0] + p[1];
	if (i == 1) return p[0] + p[2];
	if (i == 2) return p[1] + p[2];
	if (i == 3) return p[0] - p[1];
	if (i == 4) return p[0] - p[2];
	if (i == 5) return p[1] - p[2];
	return 0;
}

void kDOP18::getDistances(const Eigen::Vector3d& p, double& d3, double& d4, double& d5, double& d6, double& d7, double& d8) const
{
	d3 = p[0] + p[1];
	d4 = p[0] + p[2];
	d5 = p[1] + p[2];
	d6 = p[0] - p[1];
	d7 = p[0] - p[2];
	d8 = p[1] - p[2];
}

void kDOP18::getDistances(const Eigen::Vector3d& p, double d[]) const
{
	d[0] = p[0] + p[1];
	d[1] = p[0] + p[2];
	d[2] = p[1] + p[2];
	d[3] = p[0] - p[1];
	d[4] = p[0] - p[2];
	d[5] = p[1] - p[2];
}


bool kDOP18::overlaps(const kDOP18& b) const
{
	for (size_t i = 0; i < 9; i++) {
		if (dist[i] > b.dist[i + 9]) return false;
		if (dist[i + 9] < b.dist[i]) return false;
	}

	return true;
}


bool kDOP18::overlaps(const kDOP18& b, kDOP18& ret) const
{
	if (!overlaps(b))
		return false;

	for (size_t i = 0; i < 9; i++) {
		ret.dist[i] = MAX(dist[i], b.dist[i]);
		ret.dist[i + 9] = MIN(dist[i + 9], b.dist[i + 9]);
	}
	return true;
}


bool kDOP18::inside(const double p[]) const
{
	for (size_t i = 0; i < 3; i++) {
		if (p[i] < dist[i] || p[i] > dist[i + 9])
			return false;
	}

	double d[6]{};

	getDistances(p, d);

	for (size_t i = 3; i < 9; i++) {
		if (d[i - 3] < dist[i] || d[i - 3] > dist[i + 9])
			return false;
	}

	return true;
}


void kDOP18::empty() {
	for (size_t i = 0; i < 9; i++) {
		dist[i] = FLT_MAX;
		dist[i + 9] = -FLT_MAX;
	}
}


Eigen::Vector3d kDOP18::center() const {

	return Eigen::Vector3d((dist[0] + dist[9]) * 0.5, (dist[1] + dist[10]) * 0.5, (dist[2] + dist[11]) * 0.5);
}

DeformBVHNode::DeformBVHNode(DeformBVHNode* parent, Face* face, kDOP18* tri_boxes, Eigen::Vector3d tri_centers[])
{
	this->left = this->right = nullptr;
	this->parent = parent;
	this->face = face;
	this->box = tri_boxes[face->index];
	this->active = true;
}

DeformBVHNode::DeformBVHNode(DeformBVHNode* parent, Face** lst, unsigned int lst_num, kDOP18* tri_boxes, Eigen::Vector3d tri_centers[])
{

	assert(lst_num > 0);
	this->left = this->right = nullptr;
	this->parent = parent;
	this->face = nullptr;
	this->active = true;

	if (lst_num == 1) {
		this->face = lst[0];
		this->box = tri_boxes[lst[0]->index];
	}
	else {
		for (unsigned int t = 0; t < lst_num; t++) {
			size_t i = lst[t]->index;
			this->box += tri_boxes[i];
		}

		if (lst_num == 2) { // must split it!
			this->left = new DeformBVHNode(this, lst[0], tri_boxes, tri_centers);
			this->right = new DeformBVHNode(this, lst[1], tri_boxes, tri_centers);
		}
		else {
			aap pln(box);
			unsigned int left_idx = 0, right_idx = lst_num - 1;

			for (unsigned int t = 0; t < lst_num; t++) {
				size_t i = lst[left_idx]->index;
				if (pln.inside(tri_centers[i]))
					left_idx++;
				else {// swap it
					Face* tmp = lst[left_idx];
					lst[left_idx] = lst[right_idx];
					lst[right_idx--] = tmp;
				}
			}

			int hal = lst_num / 2;
			if (left_idx == 0 || left_idx == lst_num) {
				this->left = new DeformBVHNode(this, lst, hal, tri_boxes, tri_centers);
				this->right = new DeformBVHNode(this, lst + hal, lst_num - hal, tri_boxes, tri_centers);

			}
			else {
				this->left = new DeformBVHNode(this, lst, left_idx, tri_boxes, tri_centers);
				this->right = new DeformBVHNode(this, lst + left_idx, lst_num - left_idx, tri_boxes, tri_centers);
			}
		}
	}
}


void DeformBVHNode::refit(bool is_ccd) {

	if (isLeaf()) {
		box = face_box(getFace(), is_ccd);
	}
	else {
		getLeftChild()->refit(is_ccd);
		getRightChild()->refit(is_ccd);

		box = getLeftChild()->box + getRightChild()->box;
	}

}


bool DeformBVHNode::find(Face* face)
{
	if (isLeaf())
		return getFace() == face;

	if (getLeftChild()->find(face))
		return true;

	if (getRightChild()->find(face))
		return true;

	return false;
}


DeformBVHTree::DeformBVHTree(Mesh& mdl, bool is_ccd)
{
	this->mdl = &mdl;
	this->is_ccd = is_ccd;

	if (!mdl.verts.empty())
		Construct();
	else
		root = nullptr;
}


DeformBVHTree::~DeformBVHTree()
{
	if (!root)
		return;
	delete root;
	delete[] face_buffer;
}

void DeformBVHTree::Construct()
{
	kDOP18 total;
	unsigned int count;

	unsigned int num_vtx = static_cast<int>(mdl->verts.size());
	unsigned int num_tri = static_cast<int>(mdl->faces.size());

	for (unsigned int i = 0; i < num_vtx; i++) {
		total += mdl->verts[i]->node->x;
		if (is_ccd)
			total += mdl->verts[i]->node->x0;
	}

	count = num_tri;

	kDOP18* tri_boxes = new kDOP18[count];
	Eigen::Vector3d* tri_centers = new Eigen::Vector3d[count];

	aap pln(total);

	face_buffer = new Face * [count];
	unsigned int left_idx = 0, right_idx = count;
	unsigned int tri_idx = 0;

	for (unsigned int i = 0; i < num_tri; i++) {
		tri_idx++;

		const Eigen::Vector3d p1 = mdl->faces[i]->v[0]->node->x;
		const Eigen::Vector3d p2 = mdl->faces[i]->v[1]->node->x;
		const Eigen::Vector3d p3 = mdl->faces[i]->v[2]->node->x;
		const Eigen::Vector3d pp1 = mdl->faces[i]->v[0]->node->x0;
		const Eigen::Vector3d pp2 = mdl->faces[i]->v[1]->node->x0;
		const Eigen::Vector3d pp3 = mdl->faces[i]->v[2]->node->x0;
	
		if (is_ccd) {
			tri_centers[tri_idx - 1] = (middle_xyz(p1, p2, p3) + middle_xyz(pp1, pp2, pp3)) * 0.5;
		}
		else {
			tri_centers[tri_idx - 1] = middle_xyz(p1, p2, p3);
		}

		if (pln.inside(tri_centers[tri_idx - 1])) 
			face_buffer[left_idx++] = mdl->faces[i];
		else 
			face_buffer[--right_idx] = mdl->faces[i];
			
		tri_boxes[tri_idx - 1] += p1;
		tri_boxes[tri_idx - 1] += p2;
		tri_boxes[tri_idx - 1] += p3;

		if (is_ccd) {
			tri_boxes[tri_idx - 1] += pp1;
			tri_boxes[tri_idx - 1] += pp2;
			tri_boxes[tri_idx - 1] += pp3;
		}
	}

	root = new DeformBVHNode();
	root->box = total;


	if (count == 1) {
		root->face = mdl->faces[0];
		root->left = root->right = nullptr;
	}
	else {
		if (left_idx == 0 || left_idx == count)
			left_idx = count / 2;
		root->left = new DeformBVHNode(root, face_buffer, left_idx, tri_boxes, tri_centers);
		root->right = new DeformBVHNode(root, face_buffer + left_idx, count - left_idx, tri_boxes, tri_centers);
	}

	delete[] tri_boxes;
	delete[] tri_centers;
}

double DeformBVHTree::refit()
{
	getRoot()->refit(is_ccd);

	return 0.;
}

kDOP18 DeformBVHTree::DeformBVHTree::box()
{
	return getRoot()->box;
}
