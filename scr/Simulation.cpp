#include "Simulation.h"
#include "MeshIO.h"
#include "Solver.h"
#include "Collision.h"

#include <json/json.h>
#include <fstream>
#include <ranges>


Simulation::Simulation(std::filesystem::path config_path)
{
	std::ifstream config_file {config_path};

	if (config_file.is_open()) {
		std::cout << "Load configuration file" << std::endl;
	}
	else {
		std::cerr << "Unable to open configuration file" << std::endl;
		exit(1);
	}

	Json::Reader reader;
	Json::Value root;

	auto parse_success = reader.parse(config_file, root);

	if (!parse_success) {
		std::cerr << "Unable to parse configuration file" << std::endl;
		exit(1);
	}

	std::filesystem::path cloth_mesh_path{root["Clothes"]["MeshPath"].asString()};

	if (!std::filesystem::exists(cloth_mesh_path)) {
		std::cerr << "Cloth Mesh Path does not exist" << std::endl;
		exit(1);
	}

	if (root["IncludeObstacle"].asBool()) {

		const Json::Value obstacles_json = root["Obstacles"];

		for (unsigned int i = 0; i < obstacles_json.size(); ++i) {

			std::filesystem::path obstacle_mesh_path{obstacles_json[i]["MeshPath"].asString()};

			if (!std::filesystem::exists(obstacle_mesh_path)) {
				std::cerr << "Obstacle Mesh Path does not exist" << std::endl;
				exit(1);
			}

			Obstacle obstacle;

			read_obj(obstacle_mesh_path, obstacle.mesh);

			obstacle.material.density = obstacles_json[i]["Density"].asDouble();

			obstacles.emplace_back(std::move(obstacle));

			bool is_save_obs_mesh = obstacles_json[i]["Is_Save"].asBool();

			if (is_save_obs_mesh) {
				std::filesystem::path obstacle_mesh_save_path{obstacles_json[i]["MeshSavePath"].asString()};
				
				if (!std::filesystem::exists(obstacle_mesh_save_path.remove_filename())) {
					std::cout << "Obstacle Mesh Save Path does not exist. Create Path." << std::endl;
					std::filesystem::create_directories(obstacle_mesh_save_path);
				}

				obstacle_is_save_and_pathes.emplace_back(is_save_obs_mesh, obstacle_mesh_save_path);
			}
			else {
				obstacle_is_save_and_pathes.emplace_back(is_save_obs_mesh, "");
			}

			obstacles.back().is_motion = obstacles_json[i]["Is_Motion"].asBool();

			if (obstacles.back().is_motion) {
				
				obstacles.back().motions.emplace_back(std::make_unique<Translation>(
					obstacles_json[i]["Motions"][0]["Translation"][0].asDouble(),
					obstacles_json[i]["Motions"][0]["Translation"][1].asDouble(),
					obstacles_json[i]["Motions"][0]["Translation"][2].asDouble()));

				obstacles.back().motion_start_end_step.emplace_back(
					obstacles_json[i]["Motions"][0]["Motion_Start"].asInt(), 
					obstacles_json[i]["Motions"][0]["Motion_End"].asInt());

				obstacles.back().motions.emplace_back(std::make_unique<Translation>(
					obstacles_json[i]["Motions"][1]["Translation"][0].asDouble(),
					obstacles_json[i]["Motions"][1]["Translation"][1].asDouble(),
					obstacles_json[i]["Motions"][1]["Translation"][2].asDouble()));

				obstacles.back().motion_start_end_step.emplace_back(
					obstacles_json[i]["Motions"][1]["Motion_Start"].asInt(),
					obstacles_json[i]["Motions"][1]["Motion_End"].asInt());

			}

			obstacles.back().collision_start_end_step.emplace_back(
				obstacles_json[i]["Collision_Start"].asInt(), obstacles_json[i]["Collision_End"].asInt());

			obstacles.back().is_deform = obstacles_json[i]["Is_Deform"].asBool();

			if (obstacles.back().is_deform) {
				std::filesystem::path motion_binary_path {obstacles_json[i]["Deform_Binary_Path"].asString()};
				obstacles.back().deform_binary_path = motion_binary_path;
			}

			obstacles.back().is_load_init_binary = obstacles_json[i]["Is_Load_Binary"].asBool();

			if (obstacles.back().is_load_init_binary) {
				std::filesystem::path obs_init_binary_path {obstacles_json[i]["Obs_Binary_Path"].asString()};
				obstacles.back().init_binary_path = obs_init_binary_path;

				std::filesystem::path obs_prev_binary_path {obstacles_json[i]["Prev_Obs_Binary_Path"].asString()};
				obstacles.back().prev_binary_path = obs_prev_binary_path;
			}
		
		}
	}

	mesh_save_path = std::filesystem::path{ root["Clothes"]["MeshSavePath"].asString() };

	if (!std::filesystem::exists(mesh_save_path.remove_filename())) {
		std::cout << "Mesh Save Path does not exist: Create Directory" << std::endl;
		std::filesystem::create_directories(mesh_save_path);
	}

	binary_save_path = std::filesystem::path{ root["Clothes"]["BinarySavePath"].asString()};

	if (!std::filesystem::exists(binary_save_path.remove_filename())) {
		std::cout << "Binary Save Path does not exist: Create Directory" << std::endl;
		std::filesystem::create_directories(binary_save_path);
	}

	binary_load_path = std::filesystem::path{ root["Clothes"]["BinaryLoadPath"].asString() };

	dt = root["Time_Step"].asDouble();

	step_num = root["Step_Number"].asInt();

	current_step = root["Initial_Step"].asInt();

	repulsion_thickness = root["Collision"]["RepulsionThickness"].asDouble();

	repulsion_stiffness = root["Collision"]["RepulsionStiffness"].asDouble();

	is_collision = root["Collision"]["Is_Collision"].asBool();

	is_proximity = root["Collision"]["Is_Proximity"].asBool();

	collision_thickness = root["Collision"]["CollisionThickness"].asDouble();

	cloth_mu = root["Collision"]["ClothFrictionCoeff"].asDouble();

	cloth_obs_mu = root["Collision"]["ClothObsFrictionCoeff"].asDouble();

	is_profile_time = root["Profile_Time"].asBool();

	is_profile_solver = root["Profile_Solver"].asBool();

	is_print_step = root["Print_Step"].asBool();

	is_elapse = root["Is_Elapse"].asBool();

	elapse_start = static_cast<size_t>(root["Elapse_Start"].asInt());

	elapse_end = static_cast<size_t>(root["Elapse_End"].asInt());

	elapse_duration = root["Elapse_Duration"].asDouble();

	is_save_mesh = root["Is_Save_Mesh"].asBool();

	is_save_binary = root["Is_Save_Binary"].asBool();

	is_load_binary = root["Is_Load_Binary"].asBool();

	save_mesh_per_steps = root["Save_Mesh_Per_Steps"].asInt();

	if (root["Solver_Type"].asString() == "Eigen") {
		solver_type = SolverType::EigenCG;
	}
	else if (root["Solver_Type"].asString() == "CUDA") {
		solver_type = SolverType::CUDACG;
	}
	else if(root["Solver_Type"].asString() == "GPU_CG") {
		solver_type = SolverType::GPUCG;
	}
	else if (root["Solver_Type"].asString() == "GPU_PCG") {
		solver_type = SolverType::GPUPCG;
	}
	else if (root["Solver_Type"].asString() == "CPU_PCG") {
		solver_type = SolverType::CPUPCG;
	}
	else if (root["Solver_Type"].asString() == "CPU_CG") {
		solver_type = SolverType::CPUCG;
	}

	env.gravity = Eigen::Vector3d(
		root["Environment"]["Gravity"][0].asDouble(),
		root["Environment"]["Gravity"][1].asDouble(),
		root["Environment"]["Gravity"][2].asDouble());

	env.handle_stiffness = root["Environment"]["Handle_Stiffness"].asDouble();

	Cloth cloth;

	read_obj(cloth_mesh_path, cloth.mesh);

	cloth.material.density = root["Clothes"]["Density"].asDouble();

	cloth.material.stretching[0] = root["Clothes"]["Stretching"][0].asDouble();
	cloth.material.stretching[1] = root["Clothes"]["Stretching"][1].asDouble();
	cloth.material.stretching[2] = root["Clothes"]["Stretching"][2].asDouble();
	cloth.material.stretching[3] = root["Clothes"]["Stretching"][3].asDouble();

	cloth.material.bending = root["Clothes"]["Bending"].asDouble();

	cloth.material.damping = root["Clothes"]["Damping"].asDouble();

	cloth.is_bending_plastic = root["Clothes"]["Is_Bending_Plastic"].asBool();

	cloth.is_debug_bending_plastic = root["Clothes"]["Is_Debug_Bending_Plastic"].asBool();

	cloth.is_bending_friction = root["Clothes"]["Is_Bending_Friction"].asBool();

	cloth.is_debug_bending_friction = root["Clothes"]["Is_Debug_Bending_Friction"].asBool();

	cloth.is_tensile_plastic = root["Clothes"]["Is_Tensile_Plastic"].asBool();

	cloth.is_debug_tensile_plastic = root["Clothes"]["Is_Debug_Tensile_Plastic"].asBool();

	cloth.is_tensile_friction = root["Clothes"]["Is_Tensile_Friction"].asBool();

	cloth.is_debug_tensile_friction = root["Clothes"]["Is_Debug_Tensile_Friction"].asBool();

	cloth.is_handle_on = root["Clothes"]["Is_Handle_On"].asBool();

	if (root["Clothes"]["Is_Reset_Stretch"].asBool())
		cloth.mesh.reset_stretch();

		//cloth.zero_stretch_strains();
			
	cloth.is_stable = root["Clothes"]["Stable"]["Is_Stable"].asBool();

	if (cloth.is_stable) {
		const Json::Value stable_steps = root["Clothes"]["Stable"]["Stable_Steps"];
		for (unsigned int i = 0; i < stable_steps.size(); ++i) 
			cloth.stable_steps.insert(stable_steps[i].asUInt());	
	}

	for (unsigned int i = 0; i < 3; ++i) {
		cloth.tensile_plastic_parameters.at(i).k_hardening = root["Clothes"]["Tensile_Plastic"]["k_hardening"][i].asDouble();
		cloth.tensile_plastic_parameters.at(i).k_hardening_0 = root["Clothes"]["Tensile_Plastic"]["k_hardening_0"][i].asDouble();
		cloth.tensile_plastic_parameters.at(i).tao = root["Clothes"]["Tensile_Plastic"]["tao"][i].asDouble();
		cloth.tensile_plastic_parameters.at(i).yield_ori = root["Clothes"]["Tensile_Plastic"]["yield_ori"][i].asDouble();

		cloth.tensile_friction_parameters.at(i).k = root["Clothes"]["Tensile_Friction"]["k"][i].asDouble();
		cloth.tensile_friction_parameters.at(i).tao = root["Clothes"]["Tensile_Friction"]["tao"][i].asDouble();
		cloth.tensile_friction_parameters.at(i).thres_0 = root["Clothes"]["Tensile_Friction"]["thres_0"][i].asDouble();
		cloth.tensile_friction_parameters.at(i).thres_inf = root["Clothes"]["Tensile_Friction"]["thres_inf"][i].asDouble();
	}

	cloth.bend_plastic_parameters.k_hardening = root["Clothes"]["Bend_Plastic"]["k_hardening"].asDouble();
	cloth.bend_plastic_parameters.k_hardening_0 = root["Clothes"]["Bend_Plastic"]["k_hardening_0"].asDouble();
	cloth.bend_plastic_parameters.tao = root["Clothes"]["Bend_Plastic"]["tao"].asDouble();
	cloth.bend_plastic_parameters.yield_ori = root["Clothes"]["Bend_Plastic"]["yield_ori"].asDouble();

	cloth.bend_friction_parameters.k = root["Clothes"]["Bend_Friction"]["k"].asDouble();
	cloth.bend_friction_parameters.tao = root["Clothes"]["Bend_Friction"]["tao"].asDouble();
	cloth.bend_friction_parameters.thres_0 = root["Clothes"]["Bend_Friction"]["thres_0"].asDouble();
	cloth.bend_friction_parameters.thres_inf = root["Clothes"]["Bend_Friction"]["thres_inf"].asDouble();

	clothes.emplace_back(std::move(cloth));

	config_file.close();
}

void Simulation::physics()
{
	std::vector<Mesh*> cloth_meshes, obstacle_meshes, obs_active_meshes;

	std::vector<MeshLog> mesh_logs;

	std::vector<std::vector<sparse_tri>> mass_triplets;

	auto start = std::chrono::high_resolution_clock::now();

	for (unsigned int c = 0; c < clothes.size(); ++c) {

		cloth_meshes.emplace_back(&clothes[c].mesh);

		std::vector<sparse_tri> mass_triplet;

		clothes[c].initialize_mesh_parameters();

		clothes[c].cal_mass(mass_triplet, env.gravity);

		mass_triplets.emplace_back(mass_triplet);

		if (clothes[c].is_handle_on) clothes[c].set_handles();

		if (is_save_binary) {
			mesh_logs.emplace_back(MeshLog());
			mesh_logs.back().connect_mesh(clothes[c].mesh);
		}

		if (is_load_binary) {
			if (!is_save_binary) {
				mesh_logs.emplace_back(MeshLog());
				mesh_logs.back().connect_mesh(clothes[c].mesh);
			}
			mesh_logs[c].load_mesh_binary(binary_load_path);
		}
	}

	for (Obstacle& obstacle : obstacles) {
		obstacle_meshes.emplace_back(&obstacle.mesh);

		obstacle.cal_mass();

		if (obstacle.is_load_init_binary)
			obstacle.load_init_binary(dt);
	}

	Collision collision;

	std::vector<IneqCon*> constrains;

	for (size_t s = static_cast<size_t>(current_step); s < step_num; ++s) {

		if(is_print_step) std::cout << "Step: " << s << "\n";

		obs_active_meshes.clear();

		for (unsigned int o = 0; o < obstacles.size(); ++o) {
			obstacles[o].update_state(static_cast<int>(s));
			obstacles[o].excute_motion(static_cast<int>(s), dt);

			if (obstacles[o].is_collision_active)
				obs_active_meshes.emplace_back(&obstacles[o].mesh);
		}

		if (is_proximity && is_profile_time) start = std::chrono::high_resolution_clock::now();

		if (is_proximity) {
			Proximity proximity(repulsion_thickness, repulsion_stiffness);

			std::cout << "Obstacle Meshes: " << obs_active_meshes.size() << std::endl;

			proximity.proximity_constraints_current(cloth_meshes, obs_active_meshes, cloth_mu, cloth_obs_mu, constrains);

			std::cout << "Number of Constrains: " << constrains.size() << std::endl;
		}

		if (is_proximity && is_profile_time) std::cout << "Compute Constraints Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

		for (unsigned int c = 0; c < clothes.size(); ++c) {

			const size_t num_nodes = clothes[c].mesh.nodes.size();

			std::vector<sparse_tri> MatA_Jacob;
			Eigen::VectorXd VecB(num_nodes * 3);
			VecB.setZero();

			if (clothes[c].is_handle_on) clothes[c].update_handles(static_cast<int>(s));

			if (is_profile_time) start = std::chrono::high_resolution_clock::now();

			clothes[c].cal_external_force(VecB, env.gravity, dt);
			clothes[c].cal_handle_constrains(MatA_Jacob, VecB, static_cast<int>(s), env.handle_stiffness, dt);
			clothes[c].cal_stretch(MatA_Jacob, VecB, dt);
			clothes[c].cal_bend(MatA_Jacob, VecB, dt);

			if(is_proximity && !constrains.empty()) clothes[c].cal_constraint(MatA_Jacob, VecB, constrains, dt);
			if (clothes[c].is_stable &&
				(clothes[c].stable_steps.find(static_cast<unsigned int>(s)) != clothes[c].stable_steps.end())) {
				clothes[c].stable_nodes();
			}

			if (is_profile_time) std::cout << "Calculate Force Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

			if (is_elapse && s >= elapse_start && s <= elapse_end)
				clothes[0].time_elapse(static_cast<double>(elapse_duration / (elapse_end - elapse_start + 1)));

			MatA_Jacob.insert(MatA_Jacob.end(), mass_triplets[c].begin(), mass_triplets[c].end());

			if (is_profile_time) start = std::chrono::high_resolution_clock::now();

			sparse_M Spa_MatA(num_nodes * 3, num_nodes * 3);
			Spa_MatA.setFromTriplets(MatA_Jacob.begin(), MatA_Jacob.end());

			double* delta_vel = new double[num_nodes * 3];
			for (size_t d = 0; d < (num_nodes * 3); ++d) delta_vel[d] = 0.0;
			
			if (solver_type == SolverType::CUDACG) {
				cuda_cg(num_nodes * 3, Spa_MatA.outerIndexPtr(), Spa_MatA.innerIndexPtr(), Spa_MatA.valuePtr(), delta_vel, VecB.data());
			}
			else if (solver_type == SolverType::EigenCG) {
				Eigen::Map<Eigen::VectorXd> delta_vel_vector(delta_vel, num_nodes * 3);

				Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
				cg.compute(Spa_MatA);
				delta_vel_vector = cg.solve(VecB);

				if (is_profile_solver) {
					std::cout << "#iterations:     " << cg.iterations() << std::endl;
					std::cout << "estimated error: " << cg.error() << std::endl;
				}
			}
			else if(solver_type == SolverType::GPUCG){		
				gpu_cg(num_nodes * 3, Spa_MatA.outerIndexPtr(), Spa_MatA.innerIndexPtr(), Spa_MatA.valuePtr(), delta_vel, VecB.data(), is_profile_solver);
			}
			else if (solver_type == SolverType::GPUPCG) {

				start = std::chrono::high_resolution_clock::now();
				Eigen::MatrixXd Pre_M = Spa_MatA.diagonal().cwiseInverse().asDiagonal();
				sparse_M Pre_spa_M = Pre_M.sparseView();
				std::cout << "Create Precondictionor Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

				start = std::chrono::high_resolution_clock::now();
				
				gpu_pcg(num_nodes * 3, Spa_MatA.outerIndexPtr(), Spa_MatA.innerIndexPtr(), Spa_MatA.valuePtr(),
					Pre_spa_M.outerIndexPtr(), Pre_spa_M.innerIndexPtr(), Pre_spa_M.valuePtr(),
					delta_vel, VecB.data(), is_profile_solver);
				
				std::cout << "Solving Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

			}
			else if (solver_type == SolverType::CPUCG) {
				cpu_cg(Spa_MatA, VecB, delta_vel);
			}
			else if (solver_type == SolverType::CPUPCG) {
				cpu_pcg(Spa_MatA, VecB, delta_vel);
			}
			
			if (is_profile_time) std::cout << "Solve Governing Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

			if (is_profile_time) start = std::chrono::high_resolution_clock::now();

			for (size_t n = 0; n < clothes[c].mesh.nodes.size(); ++n) {

				clothes[c].mesh.nodes[n]->x0 = clothes[c].mesh.nodes[n]->x;

				for (size_t i = 0; i < 3; ++i) {		
					clothes[c].mesh.nodes[n]->v[i] += delta_vel[n * 3 + i];
					clothes[c].mesh.nodes[n]->x[i] += (dt * clothes[c].mesh.nodes[n]->v[i]);
				}
			}

			delete[] delta_vel;

			if (!constrains.empty() && is_proximity) {
				for (unsigned int con = 0; con < constrains.size(); ++con)
					delete constrains[con];
				constrains.clear();
			}

			if (is_profile_time) std::cout << "Update Mesh Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

			if (is_profile_time) start = std::chrono::high_resolution_clock::now();

			for (unsigned int o = 0; o < obstacles.size(); ++o)
				obstacles.at(o).excute_deformation(s, 1, dt);

			if (is_profile_time) std::cout << "Deformation Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

		} // Close Clothes iteration loop

		if (is_collision && is_profile_time) start = std::chrono::high_resolution_clock::now();

		if (is_collision) collision.collision_response(cloth_meshes, obs_active_meshes, collision_thickness, dt, is_profile_time);

		if (is_collision && is_profile_time) std::cout << "Collision Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

		if (s % save_mesh_per_steps == 0) {
			
			for (unsigned int c = 0; c < clothes.size(); ++c) {

				if (is_profile_time && is_save_mesh) start = std::chrono::high_resolution_clock::now();

				if (is_save_mesh) save_obj(mesh_save_path.replace_filename(std::string("Cloth_.obj").insert(6, std::to_string(s))), clothes[c].mesh);

				if (is_profile_time && is_save_mesh) std::cout << "Save Mesh Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

				if (is_profile_time && is_save_binary) start = std::chrono::high_resolution_clock::now();

				if (is_save_binary) mesh_logs[c].save_mesh_binary(binary_save_path.replace_filename(std::string("Cloth_.bin").insert(6, std::to_string(s))));

				if (is_profile_time && is_save_binary) std::cout << "Save Binary Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;

			} // Close Save Cloth Mesh Loop

			for (unsigned int o = 0; o < obstacles.size(); ++o) {

				if (!obstacle_is_save_and_pathes.at(o).first)
					continue;

				if (!obstacles[o].is_collision_active)
					continue;

				if (is_profile_time) start = std::chrono::high_resolution_clock::now();

				save_obj(obstacle_is_save_and_pathes.at(o).second.replace_filename(std::string("Obs_.obj").insert(4, std::to_string(s))), obstacles[o].mesh);

				if (is_profile_time) std::cout << "Save Obstacle Mesh Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
			
			}// Close Save Obstacle Mesh Loop
		}

	}// Close Simulation Step Loop

}
