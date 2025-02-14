#include <execution>
#include <map>

#include "Collision.h"
#include "Optimization.h"
#include "Auglag.h"

inline int NEXT(int i)
{
    return (i < 2) ? (i + 1) : (i - 2);
}

inline int PREV(int i)
{
    return (i > 0) ? (i - 1) : (i + 2);
}

static std::map<const Node*, Eigen::Vector3d> xold;

void mark_descendants(DeformBVHNode* node, bool active);
void mark_ancestors(DeformBVHNode* node, bool active);

void mark_all_inactive(AccelStruct& acc)
{
    if (acc.root) {
        mark_descendants(acc.root, false);
    }
}

void mark_active(AccelStruct& acc, const Face* face)
{
    if (acc.root) {
        mark_ancestors(acc.leaves[face->index], true);
    }
}

void mark_descendants(DeformBVHNode* node, bool active)
{
    node->active = active;

    if (!node->isLeaf()) {
        mark_descendants(node->left, active);
        mark_descendants(node->right, active);
    }
}

void mark_ancestors(DeformBVHNode* node, bool active)
{
    node->active = active;
    if (!node->isRoot())
        mark_ancestors(node->parent, active);
}

int find_node_in_meshes(const Node* n, const std::vector<Mesh*>& meshes)
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

int find_edge_in_meshes(const Edge* e, const std::vector<Mesh*>& meshes)
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

int find_face_in_meshes(const Face* f, const std::vector<Mesh*>& meshes)
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

std::pair<bool, int> is_node_in_meshes(const Node* node, const std::vector<Mesh*>& cloth_meshes, const std::vector<Mesh*>& obs_meshes)
{
    int m = find_node_in_meshes(node, cloth_meshes);

    if (m != -1)
    {
        return std::make_pair(true, m);
    }
    else
    {
        return std::make_pair(false, find_node_in_meshes(node, obs_meshes));
    }
}

void Collision::build_node_lookup(const std::vector<Mesh*>& meshes)
{
    for (unsigned int m = 0; m < meshes.size(); ++m)
        for (size_t n = 0; n < meshes[m]->nodes.size(); ++n)
            xold.insert({ meshes[m]->nodes[n], meshes[m]->nodes[n]->x });
}

void Collision::collision_response(std::vector<Mesh*>& cloth_meshes, const std::vector<Mesh*>& obs_meshes, const double& thickness, const double& time_step, const bool is_profile_time, const unsigned int max_iter)
{
    this->cloth_meshes = cloth_meshes;
    this->obs_meshes = obs_meshes;

    xold.clear();

    auto start = std::chrono::high_resolution_clock::now();
    
    build_node_lookup(cloth_meshes);  // Get Cloth Nodes' old position
    build_node_lookup(obs_meshes);    // Get Obstacle Nodes' old position

    if(is_profile_time) std::cout << "Save xold Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    if (is_profile_time) start = std::chrono::high_resolution_clock::now();
    
    std::vector<AccelStruct*> accs = create_accel_structs(cloth_meshes, true);
    std::vector<AccelStruct*> obs_accs = create_accel_structs(obs_meshes, true);

    if (is_profile_time) std::cout << "Create BVH Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    std::vector<ImpactZone*> zones, prezones;

    unsigned int iter{ 0 };

    for (iter = 0; iter < max_iter; ++iter) {

        //std::cout << "Iter: " << iter << std::endl;

        zones.clear();  // Clear Impace t Zone

        for (ImpactZone* p : prezones) {
            ImpactZone* newp = new ImpactZone;
            *newp = *p;
            zones.push_back(newp);
        }

        for (auto p : prezones)
            if (!p->active)
                delete p;

        if (!zones.empty())
            update_active(accs, obs_accs, zones);

        if (is_profile_time) start = std::chrono::high_resolution_clock::now();

        std::vector<Impact> impacts = find_impacts(accs, obs_accs, thickness, is_profile_time);

        if (is_profile_time) std::cout << "Find Impact Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

        if (is_profile_time) start = std::chrono::high_resolution_clock::now();

        impacts = independent_impacts(impacts);

        if (is_profile_time) std::cout << "Independent Impact Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

        if (impacts.empty())
            break;

        if (is_profile_time) start = std::chrono::high_resolution_clock::now();

        add_impacts(impacts, zones);

        if (is_profile_time) std::cout << "Add Impact Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

        if (is_profile_time) start = std::chrono::high_resolution_clock::now();

//#pragma omp parallel for
        for (int z = 0; z < zones.size(); z++) {
            ImpactZone* zone = zones[z];
            if (zone->active)
                apply_inelastic_projection(zone, thickness);
        }

        if (is_profile_time) std::cout << "Collision response Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

        if (is_profile_time) start = std::chrono::high_resolution_clock::now();

        for (int a = 0; a < accs.size(); a++)
            update_accel_struct(*accs[a]);
        for (int a = 0; a < obs_accs.size(); a++)
            update_accel_struct(*obs_accs[a]);

        if (is_profile_time) std::cout << "Update BVH Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

        prezones = zones;
    }

    if (iter == max_iter) {
        std::cerr << "Collision resolution failed to converge!" << std::endl;

        exit(1);
    }

    if (is_profile_time) start = std::chrono::high_resolution_clock::now();

    // Collision Responce: Update Cloth Meshes 
    for (unsigned int m = 0; m < cloth_meshes.size(); ++m) {
        for (unsigned int n = 0; n < cloth_meshes[m]->nodes.size(); ++n) {
            
            Node* node = cloth_meshes[m]->nodes[n];

            node->x0 = node->x;
            node->v = node->v + (node->x - xold.at(node)) / time_step;
        }
        cloth_meshes[m]->update_norms();
    }

    if (is_profile_time) std::cout << "Update Mesh Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    // Clear Dynamics Memory
    for (int z = 0; z < zones.size(); z++)
        delete zones[z];

    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);

    if(!this->cloth_meshes.empty()) this->cloth_meshes.clear();
    if (!this->obs_meshes.empty()) this->obs_meshes.clear();
}

void Collision::update_active(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, const std::vector<ImpactZone*>& zones)
{

    for (int a = 0; a < accs.size(); a++) {
        mark_all_inactive(*accs[a]);
    }

    for (int z = 0; z < zones.size(); z++)
    {
        const ImpactZone* zone = zones[z];
        if (!zone->active)
            continue;
        for (int n = 0; n < zone->nodes.size(); n++)
        {           
            const Node* node = zone->nodes[n];
            std::pair<bool, int> mi = is_node_in_meshes(node, cloth_meshes, obs_meshes);
            AccelStruct* acc = (mi.first ? accs : obs_accs)[mi.second];
            for (int v = 0; v < node->verts.size(); v++)
                for (int f = 0; f < node->verts[v]->adj_faces.size(); f++) 
                    mark_active(*acc, node->verts[v]->adj_faces[f]);                 
        }
    }
}

std::vector<Impact> Collision::find_impacts(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, const double& thickness, const bool is_profile_time)
{
    if (impacts == nullptr) {
        nthreads = omp_get_max_threads();
        impacts = new std::vector<Impact>[nthreads];
        faceimpacts = new std::vector<std::pair<Face const*, Face const*>>[nthreads];
        cnt = new int[nthreads];
    }

    for (unsigned int t = 0; t < nthreads; t++) {
        impacts[t].clear();
        faceimpacts[t].clear();
        cnt[t] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for_overlapping_faces(accs, obs_accs, thickness);

    if (is_profile_time) std::cout << "for overlap Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    if (is_profile_time) start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<Face const*, Face const*>> tot_faces;

    for (unsigned int t = 0; t < nthreads; ++t) {
        tot_faces.insert(tot_faces.end(), faceimpacts[t].begin(), faceimpacts[t].end());
    }

    if (is_profile_time) std::cout << "Insert Face Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    if (is_profile_time) start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < tot_faces.size(); ++i) {
        compute_face_impacts(tot_faces[i].first, tot_faces[i].second, thickness);
    }

    if (is_profile_time) std::cout << "Compute Face Impact Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    if (is_profile_time) start = std::chrono::high_resolution_clock::now();

    std::vector<Impact> loc_impacts;

    for (unsigned int t = 0; t < nthreads; t++) {
        loc_impacts.insert(loc_impacts.end(), impacts[t].begin(), impacts[t].end());
    }

    if (is_profile_time) std::cout << "Combine Impact Vectors Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    return loc_impacts;
}

void Collision::for_overlapping_faces(DeformBVHNode* node, const double& thickness)
{
    
    if (node->isLeaf() || !node->active)
        return;
    for_overlapping_faces(node->getLeftChild(), thickness);
    for_overlapping_faces(node->getRightChild(), thickness);
    for_overlapping_faces(node->getLeftChild(), node->getRightChild(), thickness);
}

void Collision::find_face_impacts(const Face* face0, const Face* face1)
{
    int t = omp_get_thread_num();

    faceimpacts[t].emplace_back(face0, face1);
}

void Collision::for_overlapping_faces(DeformBVHNode* node0, DeformBVHNode* node1, const double& thickness)
{        
    //cnt_called++;
    
    if (!node0->active && !node1->active)
        return;
    
    if (!overlap(node0->box, node1->box, thickness))
        return;
        
    if (node0->isLeaf() && node1->isLeaf()) {
        Face* face0 = node0->getFace(),
            * face1 = node1->getFace();
        find_face_impacts(face0, face1);
    }
    else if (node0->isLeaf()) {
        for_overlapping_faces(node0, node1->getLeftChild(), thickness);
        for_overlapping_faces(node0, node1->getRightChild(), thickness);
    }
    else {
        for_overlapping_faces(node0->getLeftChild(), node1, thickness);
        for_overlapping_faces(node0->getRightChild(), node1, thickness);
    }
}

std::vector<DeformBVHNode*> collect_upper_nodes(const std::vector<AccelStruct*>& accs, int nnodes) {

    std::vector<DeformBVHNode*> nodes;
    for (int a = 0; a < accs.size(); a++)
        if (accs[a]->root)
            nodes.push_back(accs[a]->root);
    while (nodes.size() < nnodes) {
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

void Collision::for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, double thickness, bool parallel)
{
    //omp_set_num_threads(1);
    int nnodes = (int)ceil(sqrt(2 * omp_get_max_threads()));
    std::vector<DeformBVHNode*> nodes = collect_upper_nodes(accs, nnodes);
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(parallel ? omp_get_max_threads() : 1);

    //auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int n = 0; n < nodes.size(); n++) {
        for_overlapping_faces(nodes[n], thickness); // Cloth Self-Collision
        for (int m = 0; m < n; m++)
            for_overlapping_faces(nodes[n], nodes[m], thickness);   // Cloth-Cloth Collision
        for (int o = 0; o < obs_accs.size(); o++)
            if (obs_accs[o]->root)
                for_overlapping_faces(nodes[n], obs_accs[o]->root, thickness);  // Cloth-Obstacle Collision
    }

    //std::cout << "cnt_called: " << cnt_called << std::endl;

    //std::cout << "Parallel go through BVH Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

    //exit(0);

    // Obstacle-Obstacle Collision
//    nodes = collect_upper_nodes(obs_accs, nnodes);
//#pragma omp parallel for
//    for (int n = 0; n < nodes.size(); n++) {
//        for_overlapping_faces(nodes[n], thickness);
//        for (int m = 0; m < n; m++)
//            for_overlapping_faces(nodes[n], nodes[m], thickness);
//        //for (int o = 0; o < obs_accs.size(); o++)
//        //    if (obs_accs[o]->root)
//        //        for_overlapping_faces(nodes[n], obs_accs[o]->root, thickness, callback);
//    }
    //omp_set_num_threads(nthreads);
}

void Collision::compute_face_impacts(const Face* face0, const Face* face1, const double& thickness)
{
    int t = omp_get_thread_num();

    Impact impact;

    kDOP18 nb[6], eb[6], fb[2];

    for (int v = 0; v < 3; ++v) {
        nb[v] = node_box(face0->v[v]->node, true);
        nb[v + 3] = node_box(face1->v[v]->node, true);
    }

    for (int v = 0; v < 3; ++v) {
        eb[v] = nb[NEXT(v)] + nb[PREV(v)];
        eb[v + 3] = nb[NEXT(v) + 3] + nb[PREV(v) + 3];
    }

    fb[0] = nb[0] + nb[1] + nb[2];
    fb[1] = nb[3] + nb[4] + nb[5];

    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v], fb[1], thickness))
            continue;
        if (vf_collision_test(face0->v[v], face1, impact, thickness))
            impacts[t].push_back(impact);
    }

    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v + 3], fb[0], thickness))
            continue;
        if (vf_collision_test(face1->v[v], face0, impact, thickness))
            impacts[t].push_back(impact);
    }

    for (int e0 = 0; e0 < 3; e0++) {
        for (int e1 = 0; e1 < 3; e1++)
        {
            if (!overlap(eb[e0], eb[e1 + 3], thickness))
                continue;
            if (ee_collision_test(face0->adj_edges[e0], face1->adj_edges[e1], impact, thickness))
                impacts[t].push_back(impact);
        }
    }
}

bool Collision::vf_collision_test(const Vert* vert, const Face* face, Impact& impact, const double& thickness)
{
    const Node* node = vert->node;
    if (node == face->v[0]->node
        || node == face->v[1]->node
        || node == face->v[2]->node)
        return false;

    return collision_test(Impact::VF, node, face->v[0]->node, face->v[1]->node, face->v[2]->node, impact, thickness);
}

bool Collision::ee_collision_test(const Edge* edge0, const Edge* edge1, Impact& impact, const double& thickness) {
    if (edge0->nodes[0] == edge1->nodes[0] || edge0->nodes[0] == edge1->nodes[1]
        || edge0->nodes[1] == edge1->nodes[0] || edge0->nodes[1] == edge1->nodes[1])
        return false;

    return collision_test(Impact::EE, edge0->nodes[0], edge0->nodes[1], edge1->nodes[0], edge1->nodes[1], impact, thickness);
}

int solve_quadratic(double a, double b, double c, double x[2]) {

    double d = b * b - 4 * a * c;
    if (d < 0) {
        x[0] = -b / (2 * a);
        return 0;
    }
    double q = -(b + sqrt(d)) / 2;
    double q1 = -(b - sqrt(d)) / 2;
    int i = 0;
    if (abs(a) > 1e-12) {
        x[i++] = q / a;
        x[i++] = q1 / a;
    }
    else {
        x[i++] = -c / b;
    }
    if (i == 2 && x[0] > x[1])
        std::swap(x[0], x[1]);
    return i;
}

double newtons_method(double a, double b, double c, double d, double x0,
    int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        double y0 = d + x0 * (c + x0 * (b + x0 * a)),
            ddy0 = 2 * b + (x0 + init_dir * 1e-6) * (6 * a);
        x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        double y = d + x0 * (c + x0 * (b + x0 * a));
        double dy = c + x0 * (2 * b + x0 * 3 * a);
        if (dy == 0)
            return x0;
        double x1 = x0 - y / dy;
        if (abs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

std::vector<double> solve_cubic_forward_eigen(double a, double b, double c, double d)
{
    double xc[2]{ -1.0 , -1.0  };
    double x[3]{ -1.0 , -1.0 , -1.0 };
    int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
    if (ncrit == 0) {
        x[0] = newtons_method(a, b, c, d, xc[0], 0);
        return std::vector<double>{ x[0] };
    }
    else if (ncrit == 1) {// cubic is actually quadratic
        int nsol = solve_quadratic(b, c, d, x);
        return std::vector<double>(x, x + nsol);
    }
    else {
        double yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
                        d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
        int i = 0;
        if (yc[0] * a >= 0)
            x[i++] = newtons_method(a, b, c, d, xc[0], -1);
        if (yc[0] * yc[1] <= 0) {
            int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
            x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
        }
        if (yc[1] * a <= 0)
            x[i++] = newtons_method(a, b, c, d, xc[1], 1);
        return std::vector<double>(x, x + i);
    }
}

bool Collision::collision_test(Impact::Type type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact, const double& thickness)
{
    int t0 = omp_get_thread_num();
    ++cnt[t0];
    impact.type = type;

    const Eigen::Vector3d x0_pos = node0->x;
    const Eigen::Vector3d x0_pos_prev = node0->x0;
    const Eigen::Vector3d x1_pos = node1->x;
    const Eigen::Vector3d x1_pos_prev = node1->x0;
    const Eigen::Vector3d x2_pos = node2->x;
    const Eigen::Vector3d x2_pos_prev = node2->x0;
    const Eigen::Vector3d x3_pos = node3->x;
    const Eigen::Vector3d x3_pos_prev = node3->x0;

    Eigen::Vector3d x1 = x1_pos_prev - x0_pos_prev;
    Eigen::Vector3d x2 = x2_pos_prev - x0_pos_prev;
    Eigen::Vector3d x3 = x3_pos_prev - x0_pos_prev;
    Eigen::Vector3d v0 = x0_pos - x0_pos_prev;
    Eigen::Vector3d v1 = (x1_pos - x1_pos_prev) - v0;
    Eigen::Vector3d v2 = (x2_pos - x2_pos_prev) - v0;
    Eigen::Vector3d v3 = (x3_pos - x3_pos_prev) - v0;

    double a0 = x1.dot(x2.cross(x3));
    double a1 = v1.dot(x2.cross(x3)) + x1.dot(v2.cross(x3)) + x1.dot(x2.cross(v3));
    double a2 = x1.dot(v2.cross(v3)) + v1.dot(x2.cross(v3)) + v1.dot(v2.cross(x3));
    double a3 = v1.dot(v2.cross(v3));

    std::vector<double> t = solve_cubic_forward_eigen(a3, a2, a1, a0);

    bool inside = false;
    bool over = false;

    for (int i = 0; i < t.size(); i++) {
        if (t[i] < 0.0 || t[i] > 1.0)
            continue;
        impact.t = t[i];

        Eigen::Vector3d bx1 = x1 + t[i] * v1;
        Eigen::Vector3d bx2 = x2 + t[i] * v2;
        Eigen::Vector3d bx3 = x3 + t[i] * v3;

        Eigen::Vector3d n_eigen;
        double w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;

        if (type == Impact::VF) {

            n_eigen = (bx2 - bx1).cross(bx3 - bx1);

            if (n_eigen.dot(n_eigen) < 1e-16) {
                over = true;
                continue;
            }

            n_eigen /= n_eigen.norm();

            auto h = abs(-(bx1).dot(n_eigen));

            if (h > thickness) {
                over = true;
                continue;
            }

            auto b0 = (bx2).dot((bx3).cross(n_eigen));
            auto b1 = (bx3).dot((bx1).cross(n_eigen));
            auto b2 = (bx1).dot((bx2).cross(n_eigen));

            auto sum = 1.0 / (b0 + b1 + b2);

            w0 = 1.0;
            w1 = -b0 * sum;
            w2 = -b1 * sum;
            w3 = -b2 * sum;

            inside = (std::min(std::min(-w1, -w2), -w3) >= -1e-12);

            if (!inside) {
                continue;
            }

            impact.w[0] = w0;
            impact.w[1] = w1;
            impact.w[2] = w2;
            impact.w[3] = w3;
        }
        else {

            n_eigen = bx1.cross(bx3 - bx2);

            if (n_eigen.dot(n_eigen) < 1e-16) {
                over = true;
                continue;
            }

            n_eigen /= n_eigen.norm();

            auto h = abs(-(bx2).dot(n_eigen));

            if (h > thickness) {
                over = true;
                continue;
            }

            auto a0 = (bx3 - bx1).dot((bx2 - bx1).cross(n_eigen));
            auto a1 = bx2.dot(bx3.cross(n_eigen));

            auto b0 = (bx3 - bx1).dot(bx3.cross(n_eigen));
            auto b1 = bx2.dot((bx2 - bx1).cross(n_eigen));

            auto sum_a = 1.0 / (a0 + a1);
            auto sum_b = 1.0 / (b0 + b1);

            w0 = a0 * sum_a;
            w1 = a1 * sum_a;
            w2 = -b0 * sum_b;
            w3 = -b1 * sum_b;

            bool inside = std::min(std::min(w0, w1), std::min(-w2, -w3)) >= -1e-12;

            if (!inside) {
                continue;
            }

            impact.w[0] = w0;
            impact.w[1] = w1;
            impact.w[2] = w2;
            impact.w[3] = w3;
        }

        if (n_eigen.dot(w1 * v1 + w2 * v2 + w3 * v3) > 0.0) {
            n_eigen = -n_eigen;
        }

        impact.n = n_eigen;

        impact.nodes[0] = const_cast<Node*>(node0);
        impact.nodes[1] = const_cast<Node*>(node1);
        impact.nodes[2] = const_cast<Node*>(node2);
        impact.nodes[3] = const_cast<Node*>(node3);

        return true;
    }
}

bool is_movable(const Node* n, const std::vector<Mesh*>& meshes)
{
    // Return is the given node is in the meshes of cloths
    return find_node_in_meshes(n, meshes) != -1;
}

int find_node_in_nodes(const Node* n, Node* const* ns, int num_ns)
{
    for (int i = 0; i < num_ns; ++i)
    {
        if (ns[i] == n)
            return i;
    }

    return -1;
}

int find_node_in_nodes(const Node* n, std::vector<Node*>& ns)
{
    for (int i = 0; i < ns.size(); ++i)
    {
        if (ns[i] == n)
            return i;
    }

    return -1;
}

bool is_node_in_nodes(const Node* n, Node* const* ns, int num_ns)
{
    return find_node_in_nodes(n, ns, num_ns) != -1;
}

bool conflict(const Impact& i0, const Impact& i1, const std::vector<Mesh*>& cloth_meshes)
{
    return (is_movable(i0.nodes[0], cloth_meshes) && is_node_in_nodes(i0.nodes[0], i1.nodes, 4))
        || (is_movable(i0.nodes[1], cloth_meshes) && is_node_in_nodes(i0.nodes[1], i1.nodes, 4))
        || (is_movable(i0.nodes[2], cloth_meshes) && is_node_in_nodes(i0.nodes[2], i1.nodes, 4))
        || (is_movable(i0.nodes[3], cloth_meshes) && is_node_in_nodes(i0.nodes[3], i1.nodes, 4));
}

bool operator< (const Impact& impact0, const Impact& impact1)
{
    return (impact0.t < impact1.t);
}

std::vector<Impact> Collision::independent_impacts(const std::vector<Impact>& impacts)
{
    std::vector<Impact> sorted = impacts;
    //std::sort(sorted.begin(), sorted.end());

    std::sort(std::execution::par_unseq, sorted.begin(), sorted.end());

    std::vector<Impact> indep;

    for (int e = 0; e < sorted.size(); e++) {

        const Impact& impact = sorted[e];

        bool con = false;
        for (int e1 = 0; e1 < indep.size(); e1++) {
            if (conflict(impact, indep[e1], cloth_meshes)) {
                con = true;
                //break;
            }
        }
        if (!con)
            indep.push_back(impact);
    }

    return indep;

}

ImpactZone* Collision::find_or_create_zone(const Node* node, std::vector<ImpactZone*>& zones)
{
    for (int z = 0; z < zones.size(); z++) {
        if (find_node_in_nodes(node, zones[z]->nodes) != -1) {
            return zones[z];
        }
    }

    ImpactZone* zone = new ImpactZone;

    zone->nodes.push_back(const_cast<Node*>(node));

    zones.push_back(zone);

    return zone;
}

void Collision::merge_zones(ImpactZone* zone0, ImpactZone* zone1, std::vector<ImpactZone*>& zones)
{
    if (zone0 == zone1) {
        return;
    }
    zone0->nodes.insert(zone0->nodes.end(), zone1->nodes.begin(), zone1->nodes.end());
    zone0->impacts.insert(zone0->impacts.end(), zone1->impacts.begin(), zone1->impacts.end());
    exclude(zone1, zones);
    delete zone1;
}

void Collision::exclude(const ImpactZone* z, std::vector<ImpactZone*>& zs)
{
    int i = find_zone_in_zones(z, zs);

    remove_zone_from_zones(i, zs);
}

void Collision::remove_zone_from_zones(int i, std::vector<ImpactZone*>& zs)
{
    zs[i] = zs.back();
    zs.pop_back();
}

int Collision::find_zone_in_zones(const ImpactZone* z, std::vector<ImpactZone*> zs)
{
    for (int i = 0; i < zs.size(); ++i)
    {
        if (zs[i] == z)
            return i;
    }

    return -1;
}

void Collision::add_impacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones)
{

    for (int z = 0; z < zones.size(); z++) {
        zones[z]->active = false;
    }

    for (int i = 0; i < impacts.size(); i++)
    {
        const Impact& impact = impacts[i];

        Node* node = impact.nodes[is_movable(impact.nodes[0], cloth_meshes) ? 0 : 3];

        ImpactZone* zone = find_or_create_zone(node, zones);

        for (int n = 0; n < 4; n++) {
            if (is_movable(impact.nodes[n], cloth_meshes)) {
                merge_zones(zone, find_or_create_zone(impact.nodes[n], zones), zones);
            }
        }

        zone->impacts.push_back(impact);
        zone->active = true;
    }
}

size_t get_index(const Node* node, bool is_cloth, const std::vector<Mesh*>& cloth_meshes, const std::vector<Mesh*>& obs_meshes)
{
    size_t i = 0;

    if (is_cloth)
    {
        for (size_t m = 0; m < cloth_meshes.size(); ++m)
        {
            const std::vector<Node*>& ns = cloth_meshes[m]->nodes;
            if (node->index < ns.size() && node == ns[node->index])
                return i + node->index;
            else
                i += ns.size();
        }
    }
    else
    {
        for (size_t m = 0; m < obs_meshes.size(); ++m)
        {
            const std::vector<Node*>& ns = obs_meshes[m]->nodes;
            if (node->index < ns.size() && node == ns[node->index])
                return i + node->index;
            else
                i += ns.size();
        }
    }

    return -1;
}

// Response
struct NormalOpt : public NLConOpt {
    ImpactZone* zone;
    double inv_m;
    double thickness;
    NormalOpt() : zone{nullptr}, inv_m{ 0.0 }, thickness{ 0.0 } { nvar = ncon = 0; }
    NormalOpt(ImpactZone* zone, const double& thickness) : zone{ zone }, inv_m{ 0.0 }, thickness{ thickness } {
        nvar = static_cast<int>(zone->nodes.size() * 3);
        ncon = static_cast<int>(zone->impacts.size());
        for (size_t n = 0; n < zone->nodes.size(); n++)
            inv_m += 1.0 / zone->nodes[n]->m;
        inv_m /= zone->nodes.size();
    }

    void initialize(double* x) const;
    void precompute(const double* x) const;
    double objective(const double* x) const;
    void obj_grad(const double* x, double* grad) const;
    double constraint(const double* x, int i, int& sign) const;
    void con_grad(const double* x, int i, double factor, double* grad) const;
    void finalize(const double* x) const;
};

void Collision::apply_inelastic_projection(ImpactZone* zone, const double& thickness, bool verbose)
{
    //if (!zone->active)
    //    return;
    augmented_lagrangian_method(NormalOpt(zone, thickness));
}

inline Eigen::Vector3d get_subvec(const double* x, int i) {
    return Eigen::Vector3d(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
}

inline void set_subvec(double* x, int i, const Eigen::Vector3d& xi) {
    for (int j = 0; j < 3; j++) x[i * 3 + j] = xi[j];
}

inline void add_subvec(double* x, int i, const Eigen::Vector3d& xi) {
    for (int j = 0; j < 3; j++) x[i * 3 + j] += xi[j];
}

void NormalOpt::initialize(double* x) const {
    for (int n = 0; n < (int)zone->nodes.size(); n++)
        set_subvec(x, n, zone->nodes[n]->x);
}

void NormalOpt::precompute(const double* x) const {
    for (int n = 0; n < (int)zone->nodes.size(); n++)
        zone->nodes[n]->x = get_subvec(x, n);
}

double NormalOpt::objective(const double* x) const {
    double e = 0;
    for (int n = 0; n < (int)zone->nodes.size(); n++) {
        const Node* node = zone->nodes[n];
        Eigen::Vector3d dx = node->x - xold[node];
        e += inv_m * node->m * dx.dot(dx) / 2;
    }
    return e;
}

void NormalOpt::obj_grad(const double* x, double* grad) const {
    for (int n = 0; n < (int)zone->nodes.size(); n++) {
        const Node* node = zone->nodes[n];
        Eigen::Vector3d dx = node->x - xold.at(node);
        set_subvec(grad, n, inv_m * node->m * dx);
    }
}

double NormalOpt::constraint(const double* x, int j, int& sign) const {
    sign = 1;
    double c = -thickness;
    const Impact& impact = zone->impacts[j];
    for (int n = 0; n < 4; n++)
        c += impact.w[n] * impact.n.dot(impact.nodes[n]->x);
    return c;
}

void NormalOpt::con_grad(const double* x, int j, double factor, double* grad) const {
    const Impact& impact = zone->impacts[j];
    for (int n = 0; n < 4; n++) {
        int i = find_node_in_nodes(impact.nodes[n], zone->nodes);
        if (i != -1) {
            //std::cout << "Updata Grad: " << (factor * impact.w[n] * impact.n) << std::endl;
            //std::cout << "Factor: " << factor << std::endl;
            //std::cout << "Impact w: " << impact.w[n] << std::endl;
            //std::cout << "Impact n: " << impact.n << std::endl;
            add_subvec(grad, i, factor * impact.w[n] * impact.n);
        }
            
    }
}

void NormalOpt::finalize(const double* x) const {
    precompute(x);
}
