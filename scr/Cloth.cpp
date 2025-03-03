#include "Cloth.h"

inline double fn_sign(const double& x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); }
inline double degree_to_ratio(const double& deg) { return std::numbers::pi * deg / 180.0; }

void Cloth::set_handles()
{
	for (size_t n = 0; n < mesh.nodes.size(); ++n) {

		// Simulating Trousers (pin to fixed position)

		if (mesh.nodes.at(n)->x(2) > 0.9) {

			handles.emplace_back(mesh.nodes[n]);

			handles.back().motions.emplace_back(std::make_unique<Fixed>());

			handles.back().start_end_step.emplace_back(0, 99999);
		}

		//if ((std::abs(mesh.nodes[n]->verts[0]->u(0) - 0.0) < 0.01) ||
		//	(std::abs(mesh.nodes[n]->verts[0]->u(1) - 0.0) < 0.01) ) {

		//	handles.emplace_back(mesh.nodes[n]);

		//	handles.back().motions.emplace_back(std::make_unique<Fixed>());

		//	handles.back().start_end_step.emplace_back(0, 1010);

			//handles.emplace_back(mesh.nodes[n]);

			//handles.back().is_motion = true;

			//double d_theta{ 1.5 };

			//handles.back().motions.emplace_back(std::make_unique<Rotation>(Eigen::Vector3d(0.0, 0.0, 0.0),
			//	degree_to_ratio(d_theta), 0.0, 0.0, 0.0, 0.0, degree_to_ratio(45.0)));

			//handles.back().start_end_step.emplace_back(0, 30);

			//handles.back().motions.emplace_back(std::make_unique<Fixed>());

			//handles.back().start_end_step.emplace_back(31, 40);
		//}
		//else {

		//	handles.emplace_back(mesh.nodes[n]);

		//	//handles.back().is_motion = false;

		//	handles.back().motions.emplace_back(std::make_unique<Fixed>());

		//	handles.back().start_end_step.emplace_back(0, 1010);
		//}

	}	// End Nodes loop

}

void Cloth::update_handles(const int current_step)
{
	for (Handle& h : handles) {
		for (size_t se = 0; se < h.start_end_step.size(); ++se) {
			
			if (current_step >= h.start_end_step[se].first && current_step <= h.start_end_step[se].second) {
				
				if (current_step == h.start_end_step[se].first) {
					h.renew_anchor();
				}
				
				h.update(se);
				break;
			}
		}		
	}		
}

void Cloth::renew_anchors() {
	// Handle Anchor Position -> Node's Current Position
	for (Handle& h : handles) 
		h.renew_anchor();
}

void Cloth::time_elapse(double duration) {
	for (Edge* e : mesh.edges) {
		e->friction_plastic_state.stick_t += duration;
		e->friction_plastic_state.yield_t += duration;
	}
}

void Cloth::stable_nodes() 
{
	
	for (Node* node : mesh.nodes)
		node->v.setZero();

}

void Cloth::cal_mass(std::vector<sparse_tri>& mass_triplet, const Eigen::Vector3d& gravity)
{
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

	mass_triplet.clear();

	for (size_t n = 0; n < mesh.nodes.size(); ++n)
	{
		const Node* node = mesh.nodes[n];

		int64_t idx = static_cast<int64_t>(node->index);

		for (int i = 0; i < 3; ++i) 
			mass_triplet.emplace_back(sparse_tri(idx * 3 + i, idx * 3 + i, node->m));
	}
}

void Cloth::cal_external_force(Eigen::VectorXd& Cloth_Force, const Eigen::Vector3d& gravity, const double& dt)
{
	for (size_t i = 0; i < mesh.nodes.size(); ++i) 
		for (size_t j = 0; j < 3; ++j) {
			Cloth_Force(i * 3 + j) = dt * mesh.nodes[i]->m * gravity(j);
		}		
}

double dwell_friction(FricPlasState& state, const FrictionParameter& friction_para, const double& strain, const double& dt) {

	//if (is_debug_bending_friction) std::cout << "Theta: " << theta << std::endl;

	double strain_rate = (strain - state.strain_prev) / dt;

	double strain_var = strain - state.anchor_strain;

	double strain_thres = friction_para.thres_inf - (friction_para.thres_inf - friction_para.thres_0) * std::exp(-state.stick_t / friction_para.tao);

	//if (is_debug_bending_friction) std::cout << "Stick t: " << edge->stick_t << std::endl;

	//if (is_debug_bending_friction) std::cout << "Threshold:" << strain_thres << std::endl;

	bool is_slide = std::abs(strain_var) > strain_thres;

	state.stick_t = is_slide ? 0.0 : (state.stick_t + dt);

	state.anchor_strain += ((is_slide) ? fn_sign(strain_var) * (std::abs(strain_var) - strain_thres) : 0.0);

	//strain_var = strain - state.anchor_strain;

	strain_var = is_slide ? (fn_sign(strain_var) * strain_thres) : (strain - state.anchor_strain);

	//if (is_debug_bending_friction) std::cout << "Anchor Strain: " << edge->anchor_strain;

	// Stribeck and Vary Break away

	//double anchor_strain_rate = is_slide ? ((std::abs(strain_var) - strain_thres) / dt) : 0.0;

	//double k_s = tensile_friction_parameters.at(i).C1 * std::exp(-tensile_friction_parameters.at(i).C2 * std::abs(strain_rate)) + tensile_friction_parameters.at(i).k_0;

	//double k_friction = tensile_friction_parameters.at(i).k_c + (k_s - tensile_friction_parameters.at(i).k_c) * std::exp(-std::abs(anchor_strain_rate) / tensile_friction_parameters.at(i).vs);

	state.strain_prev = strain;

	return strain_var;
}

void hardening_plastic(FricPlasState& state, const PlasticParameter& plastic_para, const double& elastic_para, const double& strain, const double& dt) {

	//if (is_debug) std::cout << "Bending Theta: " << theta << std::endl;

	//if (is_debug) std::cout << "Yield Theta: " << edge->yield_strain << std::endl;

	double elastic_strain = strain - state.plastic_strain;

	//if (is_debug) std::cout << "Elastic Theta: " << elastic_strain << std::endl;

	bool is_yield = std::abs(elastic_strain) > state.yield_strain;

	//if (is_debug) std::cout << "Is Yield: " << (is_yield ? "Yield" : "Not Yield") << std::endl;

	bool is_plastic_same_direction = (state.plastic_direction == fn_sign(elastic_strain));

	//if (is_debug) std::cout << "Is Same Direction: " << (is_plastic_same_direction ? "Same Direction" : "Diff Direction") << std::endl;

	if (is_yield) state.yield_t += dt;

	if (is_yield && !is_plastic_same_direction) state.yield_t = 0.0;

	//if (is_debug) std::cout << "Yield Time: " << edge->yield_t << std::endl;

	double k_hardening = (1.0 - (1.0 - std::exp(-state.yield_t / plastic_para.tao)) * plastic_para.k_hardening_0) * plastic_para.k_hardening;

	//if (is_debug) std::cout << "k_hardening: " << k_hardening << std::endl;

	state.plastic_strain_hardening = is_yield ?
		(state.plastic_strain_hardening + (std::abs(elastic_strain) - state.yield_strain) * elastic_para / (k_hardening + elastic_para)) :
		state.plastic_strain_hardening;

	//if (is_debug) std::cout << "plastic hardening: " << edge->plastic_strain_hardening << std::endl;

	state.plastic_strain = is_yield ? state.plastic_strain + fn_sign(elastic_strain) * (std::abs(elastic_strain) - state.yield_strain) * elastic_para / (k_hardening + elastic_para) : state.plastic_strain;

	//if (is_debug) std::cout << "plastic strain: " << edge->plastic_strain << std::endl;

	state.plastic_direction = (is_yield ? fn_sign(elastic_strain) : state.plastic_direction);

	//if (is_debug) std::cout << "plastic direction: " << edge->plastic_direction << std::endl;

	state.yield_strain = plastic_para.yield_ori + state.plastic_strain_hardening * (k_hardening / elastic_para);

	//if (is_debug) std::cout << "yield strain: " << edge->yield_strain << std::endl;
}

void Cloth::cal_stretch(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, const double& dt)
{
	std::array<double, 3> max_strain {0.0, 0.0, 0.0};
	std::array<double, 3> max_plastic_strain {0.0, 0.0, 0.0};
	std::array<double, 3> max_anchor_strain {0.0, 0.0, 0.0};
	
	for (Face* f: mesh.faces) {

		const Eigen::Vector3d& n0_pos = f->v[0]->node->x;
		const Eigen::Vector3d& n1_pos = f->v[1]->node->x;
		const Eigen::Vector3d& n2_pos = f->v[2]->node->x;

		Eigen::Matrix<double, 3, 2> Dm_w;

		Dm_w.setZero();

		Dm_w.col(0) = n1_pos - n0_pos;
		Dm_w.col(1) = n2_pos - n0_pos;

		Eigen::Matrix<double, 3, 2> F = Dm_w * f->invDm;	// Deformation Gradient: dx/dX

		Eigen::Vector4d G = 0.5 * (F.transpose() * F - Eigen::Matrix2d::Identity()).reshaped<Eigen::RowMajor>().transpose();	// Green-Lagrange Tensor

		const Eigen::Vector4d d = f->invDm.reshaped<Eigen::RowMajor>().transpose();	// 1 / dX

		Eigen::Matrix<double, 3, 9> Du, Dv;

		Du.block<3, 3>(0, 0) = (-d[0] - d[2]) * Eigen::Matrix3d::Identity();
		Du.block<3, 3>(0, 3) = d[0] * Eigen::Matrix3d::Identity();
		Du.block<3, 3>(0, 6) = d[2] * Eigen::Matrix3d::Identity();

		Dv.block<3, 3>(0, 0) = (-d[1] - d[3]) * Eigen::Matrix3d::Identity();
		Dv.block<3, 3>(0, 3) = d[1] * Eigen::Matrix3d::Identity();
		Dv.block<3, 3>(0, 6) = d[3] * Eigen::Matrix3d::Identity();

		Eigen::Matrix<double, 9, 3> Dut = Du.transpose();
		Eigen::Matrix<double, 9, 3> Dvt = Dv.transpose();

		const Eigen::Vector3d xu = F.col(0);

		const Eigen::Vector3d xv = F.col(1);

		const Eigen::Vector<double, 9> fuu = Dut * xu;

		const Eigen::Vector<double, 9> fvv = Dvt * xv;

		const Eigen::Vector<double, 9> fuv = Dut * xv + Dvt * xu;

		Eigen::Matrix<double, 9, 9> DutDu = Dut * Du;

		Eigen::Matrix<double, 9, 9> DvtDv = Dvt * Dv;

		//std::array<double, 3> elastic_strain {G(0) - f->rest_strain.at(0), G(3) - f->rest_strain.at(1), G(1) - f->rest_strain.at(2)};
		
		std::array<double, 3> elastic_strain {G(0) , G(3) , G(1)};

		// ** Hardening Plastic **	
		if (is_tensile_plastic) {
			std::array<double, 3> principle_stretching_stiffness {material.stretching.at(0), material.stretching.at(2), material.stretching.at(3)};

			for (unsigned int i = 0; i < 3; ++i) {
				hardening_plastic(f->friction_plastic_states.at(i), tensile_plastic_parameters.at(i), principle_stretching_stiffness.at(i), elastic_strain.at(i), dt);

				elastic_strain.at(i) -= f->friction_plastic_states.at(i).plastic_strain;
			}
		}

		const Eigen::Vector<double, 9> grad_e =
			material.stretching.at(0) * elastic_strain.at(0) * fuu +
			material.stretching.at(2) * elastic_strain.at(1) * fvv +
			material.stretching.at(1) * (elastic_strain.at(0) * fvv + elastic_strain.at(1) * fuu) +
			material.stretching.at(3) * elastic_strain.at(2) * fuv;

		std::for_each(elastic_strain.begin(), elastic_strain.end(), [](double& n) { n = (n > 0.0 ? n : 0.0); });	// pos def

		Eigen::Matrix<double, 9, 9> hess_e = 
			material.stretching.at(0) * (fuu * fuu.transpose() + elastic_strain.at(0) * DutDu) + 
			material.stretching.at(2) * (fvv * fvv.transpose() + elastic_strain.at(1) * DvtDv) + 
			material.stretching.at(1) * (fuu * fvv.transpose() + elastic_strain.at(0) * DvtDv +
			fvv * fuu.transpose() + elastic_strain.at(1) * DutDu) + 0.5 * material.stretching.at(3) * (fuv * fuv.transpose());

		Eigen::Matrix<double, 9, 9> Jaco = -hess_e * f->m_a;
		Eigen::Vector<double, 9> Force = -grad_e * f->m_a;

		// **Internal Friction**
		if (is_tensile_friction) {
			std::array<double, 3> current_strain {G(0), G(3), G(1)};
			std::array<double, 3> friction_stress {0.0, 0.0, 0.0};

			for (unsigned int i = 0; i < 3; ++i) 
				friction_stress.at(i) = tensile_friction_parameters.at(i).k * dwell_friction(f->friction_plastic_states.at(i), tensile_friction_parameters.at(i), current_strain.at(i), dt);
			
			const Eigen::Vector<double, 9> grad_fric = friction_stress.at(0) * fuu + friction_stress.at(1) * fvv + friction_stress.at(2) * fuv;
			const Eigen::Matrix<double, 9, 9>  hess_friction =
				fuu * fuu.transpose() * tensile_friction_parameters.at(0).k + DutDu * friction_stress[0] +
				fvv * fvv.transpose() * tensile_friction_parameters.at(1).k + DvtDv * friction_stress[1] +
				0.5 * fuv * fuv.transpose() * tensile_friction_parameters.at(2).k;

			Force += (-grad_fric * f->m_a);
			Jaco += (-hess_friction * f->m_a);
		}

		if ((is_tensile_friction && is_debug_tensile_friction) || 
			(is_tensile_plastic && is_debug_tensile_plastic) ) {
			if (std::abs(G(0)) > max_strain.at(0)) max_strain.at(0) = std::abs(G(0));
			if (std::abs(G(3)) > max_strain.at(1)) max_strain.at(1) = std::abs(G(3));
			if (std::abs(G(1)) > max_strain.at(2)) max_strain.at(2) = std::abs(G(1));
		}

		if (is_tensile_plastic && is_debug_tensile_plastic) {
			for (unsigned int i = 0; i < 3; ++i) {
				if (std::abs(f->friction_plastic_states.at(i).plastic_strain) > max_plastic_strain.at(i)) {
					max_plastic_strain.at(i) = std::abs(f->friction_plastic_states.at(i).plastic_strain);
				}
			}
		}

		if (is_tensile_friction && is_debug_tensile_friction) {
			for (unsigned int i = 0; i < 3; ++i) {
				if (std::abs(f->friction_plastic_states.at(i).anchor_strain) > max_anchor_strain.at(i)) {
					max_anchor_strain.at(i) = std::abs(f->friction_plastic_states.at(i).anchor_strain);
				}
			}
		}

		double vel[9]{};

		for (size_t n = 0; n < 3; ++n)
			for (size_t m = 0; m < 3; ++m) {
				vel[n * 3 + m] = f->v[n]->node->v(m);
			}
				
		Eigen::Map<Eigen::Vector<double, 9>> Cloth_Velocity(vel);

		Force = dt * (Force + (dt + material.damping) * (Jaco * Cloth_Velocity));

		for (size_t i = 0; i < 3; ++i) {
			for (size_t i_e = 0; i_e < 3; ++i_e) 
				Cloth_Force(f->v[i]->node->index * 3 + i_e) += Force(i * 3 + i_e);

			for (size_t j = 0; j < 3; ++j) {
				for (size_t k = 0; k < 3; ++k)
					for (size_t l = 0; l < 3; ++l) {
						Cloth_Jacob.emplace_back(sparse_tri(
							f->v[i]->node->index * 3 + k, 
							f->v[j]->node->index * 3 + l,
							-dt * (dt + material.damping) * Jaco(i * 3 + k, j * 3 + l)));
					}
			}
		}
	}	// End Loop Faces

	if ((is_tensile_friction && is_debug_tensile_friction) ||
		(is_tensile_plastic && is_debug_tensile_plastic)) {
		std::cout << "Maximum Strain u: " << max_strain.at(0) << "\n" 
			<< "Maximum Strain v: " << max_strain.at(1) << "\n"
			<< "Maximum Strain shear: " << max_strain.at(2) << std::endl;
	}

	if (is_tensile_friction && is_debug_tensile_friction) {
		std::cout << "Maximum Anchor Strain u: " << max_anchor_strain.at(0) << "\n"
			<< "Maximum Anchor Strain v: " << max_anchor_strain.at(1) << "\n" 
			<< "Maximum Anchor Strain shear: " << max_anchor_strain.at(2) << std::endl;
	}

	if (is_tensile_plastic && is_debug_tensile_plastic) {
		std::cout << "Maximum Plastic Strain u: " << max_plastic_strain.at(0) << "\n" 
			<< "Maximum Plastic Strain v: " << max_plastic_strain.at(1) << "\n"
			<< "Maximum Plastic Strain shear: " << max_plastic_strain.at(2) << std::endl;
	}

}

void Cloth::initialize_mesh_parameters()
{

#pragma omp parallel for
	for (int e = 0; e < mesh.edges.size(); ++e) {
		Edge* edge = mesh.edges.at(e);
		// Initialize Plastic Parameters
		edge->friction_plastic_state.yield_t = 0.0;
		edge->friction_plastic_state.yield_strain = bend_plastic_parameters.yield_ori;
		edge->friction_plastic_state.plastic_direction = 0.0;

		edge->friction_plastic_state.plastic_strain = 0.0;
		edge->friction_plastic_state.plastic_strain_hardening = 0.0;

		// Initialize Friction Parameters
		edge->friction_plastic_state.stick_t = 0.0;
		edge->friction_plastic_state.anchor_strain = 0.0;
		edge->friction_plastic_state.strain_prev = 0.0;
	}

#pragma omp parallel for
	for (int f = 0; f < mesh.faces.size(); ++f) {
		Face* face = mesh.faces.at(f);

		for (int i = 0; i < 3; ++i) {
			// Initialize Plastic Parameters
			face->friction_plastic_states.at(i).yield_t = 0.0;
			face->friction_plastic_states.at(i).yield_strain = tensile_plastic_parameters.at(i).yield_ori;
			face->friction_plastic_states.at(i).plastic_direction = 0.0;

			face->friction_plastic_states.at(i).plastic_strain = 0.0;
			face->friction_plastic_states.at(i).plastic_strain_hardening = 0.0;
			// Initialize Friction Parameters
			face->friction_plastic_states.at(i).stick_t = 0.0;
			face->friction_plastic_states.at(i).anchor_strain = 0.0;
			face->friction_plastic_states.at(i).strain_prev = 0.0;
		}
	}
}

void Cloth::zero_stretch_strains()
{
#pragma omp parallel for
	for (int f = 0; f < mesh.faces.size(); ++f) {

		Face* face = mesh.faces.at(f);

		const Eigen::Vector3d& n0_pos = face->v[0]->node->x;
		const Eigen::Vector3d& n1_pos = face->v[1]->node->x;
		const Eigen::Vector3d& n2_pos = face->v[2]->node->x;

		Eigen::Matrix<double, 3, 2> Dm_w;

		Dm_w.setZero();

		Dm_w.col(0) = n1_pos - n0_pos;
		Dm_w.col(1) = n2_pos - n0_pos;

		Eigen::Matrix<double, 3, 2> F = Dm_w * face->invDm;	// Deformation Gradient: dx/dX

		Eigen::Vector4d G = 0.5 * (F.transpose() * F - Eigen::Matrix2d::Identity()).reshaped<Eigen::RowMajor>().transpose();

		face->rest_strain.at(0) = G(0);
		face->rest_strain.at(1) = G(3);
		face->rest_strain.at(2) = G(1);
	}
}

void Cloth::zero_bend_strains() 
{

#pragma omp parallel for
	for (int e = 0; e < mesh.edges.size(); ++e) {
		
		Edge* edge = mesh.edges.at(e);

		const Face* face0 = edge->adj_faces[0];
		const Face* face1 = edge->adj_faces[1];

		if (face0 == nullptr || face1 == nullptr)
			continue;

		edge->rest_theta = edge->dihedral_angle();
	}

}

void Cloth::cal_bend(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, const double& dt)
{
	double max_theta{ 0.0 }, max_plastic_strain{ 0.0 }, max_anchor_strain {0.0};
	
	for (Edge* edge: mesh.edges) {
		
		const Face* face0 = edge->adj_faces[0];
		const Face* face1 = edge->adj_faces[1];

		if (face0 == nullptr || face1 == nullptr)
			continue;

		double theta = edge->dihedral_angle();
		double a = face0->m_a + face1->m_a;

		Eigen::Vector3d x0 = edge->nodes[0]->x;
		Eigen::Vector3d x1 = edge->nodes[1]->x;

		Eigen::Vector3d x2 = edge_opp_vert(edge, 0)->node->x;
		Eigen::Vector3d x3 = edge_opp_vert(edge, 1)->node->x;

		Eigen::Vector3d e = x1 - x0;
		double dote = e.dot(e);
		double norme = std::sqrt(dote);

		double t0 = e.dot(x2 - x0) / dote;
		double t1 = e.dot(x3 - x0) / dote;

		double h0 = std::max((x2 - x0 - e * t0).norm(), norme);
		double h1 = std::max((x3 - x0 - e * t1).norm(), norme);

		Eigen::Vector3d n0 = face0->face_normal() / h0;
		Eigen::Vector3d n1 = face1->face_normal() / h1;

		Eigen::Matrix<double, 4, 2> wf { {t0 - 1.0, t1 - 1.0}, { -t0, -t1 }, { 1.0, 0.0 }, { 0.0, 1.0 } };
		
		Eigen::Matrix<double, 2, 3> n0_n1;
		n0_n1.row(0) = n0;
		n0_n1.row(1) = n1;

		Eigen::Vector<double, 12> dtheta = (wf * n0_n1).reshaped<Eigen::RowMajor>().transpose();

		// ** Hardening Plastic **
		
		//if (is_bending_plastic) hardening_plastic(edge, theta, dt, edge->index == 14551);		// Print Debug Information by selecting edge index

		if (is_bending_plastic) hardening_plastic(edge->friction_plastic_state, bend_plastic_parameters, material.bending, theta, dt);

		Eigen::Matrix<double, 12, 12> Jaco = -material.bending * ((edge->l * edge->l) / a) * (dtheta * dtheta.transpose());	// Jacobian (Find a bug: the bending stiffness in the paper should be devided by 4)

		Eigen::Vector<double, 12> Force = -material.bending * ((edge->l * edge->l) / a) * (theta - edge->friction_plastic_state.plastic_strain) * dtheta;	// Force (Find a bug: the bending stiffness in the paper should be devided by 4)

		// ** Dwell Friction **

		if (is_bending_friction) {
			double strain_var = dwell_friction(edge->friction_plastic_state, bend_friction_parameters, theta, dt);

			Jaco += -(bend_friction_parameters.k * ((edge->l * edge->l) / a) * (dtheta * dtheta.transpose())); // Find a bug: the bending friction stiffness in the paper should be devided by 4
			Force += -(bend_friction_parameters.k * ((edge->l * edge->l) / a) * strain_var) * dtheta; 	// Find a bug: the bending friction stiffness in the paper should be devided by 4
		}

		if ((is_bending_plastic && is_debug_bending_plastic) ||
			(is_bending_friction && is_debug_bending_friction))
			if (std::abs(theta) > max_theta) max_theta = std::abs(theta);

		if (is_bending_plastic && is_debug_bending_plastic)
			if (std::abs(edge->friction_plastic_state.plastic_strain) > max_plastic_strain) max_plastic_strain = std::abs(edge->friction_plastic_state.plastic_strain);

		if (is_bending_friction && is_debug_bending_friction)
			if (std::abs(edge->friction_plastic_state.anchor_strain) > max_anchor_strain) max_anchor_strain = std::abs(edge->friction_plastic_state.anchor_strain);


		std::array<const Node*, 4> nodes { 
			edge->nodes[0], 
			edge->nodes[1], 
			edge_opp_vert(edge, 0)->node,
			edge_opp_vert(edge, 1)->node };

		double vel[12] {};

		for (size_t n = 0; n < 4; ++n)
			for (size_t m = 0; m < 3; ++m) {
				vel[n * 3 + m] = nodes[n]->v(m);
			}

		Eigen::Map<Eigen::Vector<double, 12>> Cloth_Velocity(vel);

		Force = dt * (Force + (dt + material.damping) * (Jaco * Cloth_Velocity));

		for (size_t i = 0; i < 4; ++i) {
			for (size_t i_e = 0; i_e < 3; ++i_e)
				Cloth_Force(nodes[i]->index * 3 + i_e) += Force(i * 3 + i_e);
			
			for (size_t j = 0; j < 4; ++j) {

				for (size_t k = 0; k < 3; ++k)
					for (size_t l = 0; l < 3; ++l) {
						Cloth_Jacob.emplace_back(sparse_tri(
							nodes[i]->index * 3 + k,
							nodes[j]->index * 3 + l,
							-dt * (dt + material.damping) * Jaco(i * 3 + k, j * 3 + l)));
					}
			}
		}

	}// End Edge loop

	if ((is_bending_plastic && is_debug_bending_plastic) ||
		(is_bending_friction && is_debug_bending_friction)) {
		std::cout << "Maximum Theta: " << max_theta << std::endl;
	}
		
	if (is_bending_friction && is_debug_bending_friction) {
		std::cout << "Maximum Anchor Strain: " << max_anchor_strain << std::endl;
	}
		
	if (is_bending_plastic && is_debug_bending_plastic)
		std::cout << "Maximum Plastic Strain: " << max_plastic_strain << std::endl;	

}

void Cloth::cal_constraint(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, std::vector<IneqCon*>& cons, const double& dt)
{	
	for (int c = 0; c < cons.size(); ++c) {

		// Repulsion Force
		double value = cons[c]->value();
		double g = cons[c]->energy_grad(value);
		double h = cons[c]->energy_hess(value);
		std::map<Node*, Eigen::Vector3d> grad = cons[c]->gradient();

		for (auto const& mg_i : grad) {

			Node* node_i = mg_i.first;

			if (node_i->index < mesh.nodes.size() && mesh.nodes[node_i->index] == node_i) {	

				Eigen::Vector3d Force = -dt * (g + dt * h * mg_i.second.dot(node_i->v)) * mg_i.second;

				for (size_t i = 0; i < 3; ++i)
					Cloth_Force(node_i->index * 3 + i) += Force(i);

				for (auto const& mg_j : grad) {

					Node* node_j = mg_j.first;

					if (node_j->index < mesh.nodes.size() && mesh.nodes[node_j->index] == node_j) {

						Eigen::Matrix3d Jaco_loc = dt * dt * h * (mg_i.second * mg_j.second.transpose());

						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {

								Cloth_Jacob.emplace_back(sparse_tri(
									node_i->index * 3 + i,
									node_j->index * 3 + j,
									Jaco_loc(i, j) ));
							}
						}
					}
				}// End Repulsion Jaco Loop
			}

		}	// End Repulsion Loop

		//	Friction Force and Jacobian

		std::map<std::pair<Node*, Node*>, Eigen::Matrix3d> jac;
		std::map<Node*, Eigen::Vector3d> force = cons[c]->friction(dt, jac);

		for (auto& fric_f : force) {

			const Node* fric_node = fric_f.first;

			if (fric_node->index < mesh.nodes.size() && mesh.nodes[fric_node->index] == fric_node) {
				
				Eigen::Vector3d Force = fric_f.second * dt;

				for (size_t i = 0; i < 3; ++i)
					Cloth_Force(fric_node->index * 3 + i) += Force(i);
	
			}
		}// End Friction Force Loop

		for (const auto& fric_jac : jac) {

			const Node* fric_node_i = fric_jac.first.first;
			const Node* fric_node_j = fric_jac.first.second;

			if ((fric_node_i->index < mesh.nodes.size() && mesh.nodes[fric_node_i->index] == fric_node_i) &&
				(fric_node_j->index < mesh.nodes.size() && mesh.nodes[fric_node_j->index] == fric_node_j)) {

				Eigen::Matrix3d Jaco_loc = -dt * fric_jac.second;

				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {

						Cloth_Jacob.emplace_back(sparse_tri(
							fric_node_i->index * 3 + i,
							fric_node_j->index * 3 + j,
							Jaco_loc(i, j)));

					}
				}
			}

		}	// End Friction Jac Loop

	} // End Loop Constraints
}

void Cloth::cal_handle_constrains(std::vector<sparse_tri>& Cloth_Jacob, Eigen::VectorXd& Cloth_Force, const int current_step, const double& stiffness, const double& dt)
{	
	for (const Handle& h : handles) {

		bool is_active{ false };

		for (auto& se : h.start_end_step) {
			if (current_step >= se.first && current_step <= se.second) {
				is_active = true;
				break;
			}
		}

		if (!is_active)
			continue;

		Eigen::Vector3d Force = -stiffness * (h.node->x - h.anchor_pos);

		Force = dt * (Force + dt * (-stiffness * h.node->v));

		for (size_t i = 0; i < 3; ++i) {
			Cloth_Force(h.node->index * 3 + i) += Force(i);

			Cloth_Jacob.emplace_back(sparse_tri(
				h.node->index * 3 + i,
				h.node->index * 3 + i,
				dt * dt * stiffness));
		}
	}
}
