#include "Obstacle.h"
#include "MeshIO.h"

void Obstacle::cal_mass() {
	for (size_t v = 0; v < mesh.verts.size(); ++v)
		mesh.verts[v]->m = 0.0;

	for (size_t n = 0; n < mesh.nodes.size(); ++n)
		mesh.nodes[n]->m = 0.0;

	for (size_t f = 0; f < mesh.faces.size(); ++f) {

		Face* face = mesh.faces[f];

		face->m = face->m_a * material.density;

		for (int v = 0; v < 3; ++v) {
			face->v[v]->m += face->m / 3.0;
			face->v[v]->node->m += face->m / 3.0;
		}
	}
}

void Obstacle::load_init_binary(const double dt)
{
	load_obs_binary(prev_binary_path, mesh);

	mesh.update_x0();
	
	load_obs_binary(init_binary_path, mesh);

	for (Node* node : mesh.nodes)
		node->v = (node->x - node->x0) / dt;

	mesh.update_norms();
}

void Obstacle::excute_motion(const int& current_step, const double& dt)
{
	if (is_motion) {
		
		for (size_t se = 0; se < motion_start_end_step.size(); ++se) {

			if (current_step >= motion_start_end_step[se].first && current_step <= motion_start_end_step[se].second) {

				for (Node* node : mesh.nodes) {
					node->x0 = node->x;
					motions.at(se)->Move(node->x);
					node->v = (node->x - node->x0) / dt;
				}

				mesh.update_norms();
			}
		}
	}
}

void Obstacle::excute_deformation(const int& current_step, const int& offset, const double& dt)
{
	if (is_deform) {

		mesh.update_x0();	// Current Position to Previous Position
		
		deform_binary_path.replace_filename(std::format("body_{}.dat", current_step + offset));

		std::cout << "Read Motion Mesh Binary: " << deform_binary_path.filename() << std::endl;

		load_obs_binary(deform_binary_path, mesh);	// Load Binary 

		for (unsigned int n = 0; n < mesh.nodes.size(); ++n) 
			mesh.nodes.at(n)->v = (mesh.nodes.at(n)->x - mesh.nodes.at(n)->x0) / dt;

		mesh.update_norms();	// Update Mesh Normals
	}
}

void Obstacle::update_state(const int current_step) {
	// Check: Is Collision Active
	for (size_t se = 0; se < collision_start_end_step.size(); ++se) {
		if (current_step >= collision_start_end_step[se].first && current_step <= collision_start_end_step[se].second)
			is_collision_active = true;
		else
			is_collision_active = false;
	}
}
