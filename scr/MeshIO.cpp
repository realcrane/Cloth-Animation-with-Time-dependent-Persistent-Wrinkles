#include "MeshIO.h"

#include <fstream>
#include <sstream>

static const std::array<std::string, 3> PROCESS_PRIMARY = { "v", "vt", "f" };

void read_obj(const std::filesystem::path& filename, Mesh& mesh)
{
	std::ifstream obj_file{ filename };

	std::string line;

	while (std::getline(obj_file, line))
	{

		std::vector<std::string> tokens;

		auto position = line.find(" ");

		auto primary_type = line.substr(0, position);

		auto find_result = std::find(PROCESS_PRIMARY.begin(), PROCESS_PRIMARY.end(), primary_type);

		if (find_result != PROCESS_PRIMARY.end())
		{
			std::string token;

			if (primary_type == "v")
			{
				std::stringstream ss{ line };

				Eigen::Vector3d Position = Eigen::Vector3d::Zero();

				size_t token_cnt = 0;

				while (std::getline(ss, token, ' '))
				{
					token_cnt++;

					if (token_cnt > 1)
						Position[token_cnt - 2] = std::stof(token);
				}

				mesh.add(new Node(Position, Eigen::Vector3d::Zero()));

			}
			else if (primary_type == "vt")
			{
				std::stringstream ss{ line };

				Eigen::Vector3d UV = Eigen::Vector3d::Zero();

				size_t token_cnt = 0;

				while (std::getline(ss, token, ' '))
				{
					token_cnt++;

					if (token_cnt > 1)
						UV[token_cnt - 2] = std::stof(token);

				}

				mesh.add(new Vert(UV));

			}
			else if (primary_type == "f")
			{
				std::vector<Node*> nodes;
				std::vector<Vert*> verts;

				std::stringstream ss{ line };

				int token_cnt = 0;

				while (std::getline(ss, token, ' '))
				{

					token_cnt++;

					if (token_cnt > 1)
					{
						size_t deliPos = token.find("/");	// Find the Position of Delimiter
						size_t token_len = token.length();	// The Length of the read token

						std::string nodeIdx = token.substr(0, deliPos);		// The index of "v" in the .obj file
						std::string vertIdx = token.substr(deliPos + 1, token_len);	// The index of "vt" in the .obj file

						int node_idx = std::stoi(nodeIdx) - 1;
						int vert_idx = std::stoi(vertIdx) - 1;

						nodes.push_back(mesh.nodes[node_idx]);
						verts.push_back(mesh.verts[vert_idx]);
					}
				}

				// Connect Verts and Nodes
				for (size_t v = 0; v < verts.size(); ++v)
				{
					// let the member node in vert points to the node
					verts[v]->node = nodes[v];
					// let the add the vert to the verts of node class
					auto find_vert = find(nodes[v]->verts.cbegin(), nodes[v]->verts.cend(), verts[v]);
					// add only if the vert does not exits in the node's verts vertor
					if (find_vert == nodes[v]->verts.cend())
						nodes[v]->verts.push_back(verts[v]);
				}

				mesh.add(new Face(verts[0], verts[1], verts[2]));

			}
		}
	}

	mesh.compute_ms_data();
	mesh.compute_ws_data();
}

void save_obj(const std::filesystem::path& filename, Mesh& mesh)
{
	std::fstream file(filename, std::fstream::out);

	for (size_t v = 0; v < mesh.verts.size(); ++v)
	{
		const Vert* vert = mesh.verts[v];

		file << "vt " << vert->u[0] << " " << vert->u[1] << "\n";
	}

	for (size_t n = 0; n < mesh.nodes.size(); ++n)
	{
		const Node* node = mesh.nodes[n];

		file << "v " << node->x[0] << " "
			<< node->x[1] << " "
			<< node->x[2] << "\n";
	}

	for (size_t f = 0; f < mesh.faces.size(); ++f)
	{
		const Face* face = mesh.faces[f];

		file << "f " << face->v[0]->node->index + 1 << "/" << face->v[0]->index + 1 << "/" << face->v[0]->index + 1
			<< " " << face->v[1]->node->index + 1 << "/" << face->v[1]->index + 1 << "/" << face->v[0]->index + 1
			<< " " << face->v[2]->node->index + 1 << "/" << face->v[2]->index + 1 << "/" << face->v[0]->index + 1 << "\n";
	}

	file.close();
}


void write_vt_to_stringstream(std::basic_ostringstream<char>& ss, const Mesh& mesh) {
	for (const Vert* vert : mesh.verts)
		ss << "vt " << vert->u[0] << " " << vert->u[1] << "\n";
}

void write_v_to_stringstream(std::basic_ostringstream<char>& ss, const Mesh& mesh) {
	for (const Node* node : mesh.nodes) {
		ss << "v " << node->x[0] << " " << node->x[1] << " " << node->x[2] << "\n";
	}
}

void write_f_to_stringstream(std::basic_ostringstream<char>& ss, const Mesh& mesh) {
	for (const Face* face : mesh.faces) {
		ss << "f " << face->v[0]->node->index + 1 << "/" << face->v[0]->index + 1 << "/" << face->v[0]->index + 1
			<< " " << face->v[1]->node->index + 1 << "/" << face->v[1]->index + 1 << "/" << face->v[0]->index + 1
			<< " " << face->v[2]->node->index + 1 << "/" << face->v[2]->index + 1 << "/" << face->v[0]->index + 1 << "\n";
	}
}

void save_obj_efficient(const std::filesystem::path& filename, const Mesh& mesh)
{
	std::basic_ostringstream<char> ss_vt, ss_v, ss_f;

	std::array<std::thread, 3> threads;

	threads[0] = std::thread(write_vt_to_stringstream, std::ref(ss_vt), std::ref(mesh));
	threads[1] = std::thread(write_v_to_stringstream, std::ref(ss_v), std::ref(mesh));
	threads[2] = std::thread(write_f_to_stringstream, std::ref(ss_f), std::ref(mesh));

	for (auto& thread : threads)
		thread.join();

	std::fstream file(filename, std::fstream::out | std::ios::binary);
	file << ss_vt.str() << ss_v.str() << ss_f.str();
	file.close();
}

void MeshLog::connect_mesh(Mesh& mesh)
{
	// Nodes
	nodes_num = mesh.nodes.size();

	pos.resize(nodes_num);
	pos_0.resize(nodes_num);
	vel.resize(nodes_num);
	acceleration.resize(nodes_num);
	
	for (int n = 0; n < nodes_num;  ++n) {
		Node* node = mesh.nodes[n];
		pos.at(n) = &node->x;
		pos_0.at(n) = &node->x0;
		vel.at(n) = &node->v;
		acceleration.at(n) = &node->acceleration;
	}

	// Edges
	edges_num = mesh.edges.size();

	bend_yield_t.resize(edges_num);
	bend_yield_strain.resize(edges_num);
	bend_plastic_direction.resize(edges_num);

	bend_plastic_strain.resize(edges_num);
	bend_plastic_strain_hardening.resize(edges_num);

	bend_stick_t.resize(edges_num);
	bend_anchor_strain.resize(edges_num);
	bend_strain_prev.resize(edges_num);

	for (int e = 0; e < edges_num; ++e) {
		
		Edge* edge = mesh.edges[e];
		
		bend_yield_t.at(e) = &edge->friction_plastic_state.yield_t;
		bend_yield_strain.at(e) = &edge->friction_plastic_state.yield_strain;
		bend_plastic_direction.at(e) = &edge->friction_plastic_state.plastic_direction;

		bend_plastic_strain.at(e) = &edge->friction_plastic_state.plastic_strain;
		bend_plastic_strain_hardening.at(e) = &edge->friction_plastic_state.plastic_strain_hardening;

		bend_stick_t.at(e) = &edge->friction_plastic_state.stick_t;
		bend_anchor_strain.at(e) = &edge->friction_plastic_state.anchor_strain;
		bend_strain_prev.at(e) = &edge->friction_plastic_state.strain_prev;
	}

	// Faces
	faces_num = mesh.faces.size();
	
	stretch_yield_t.resize(faces_num*3);
	stretch_yield_strain.resize(faces_num*3);
	stretch_plastic_direction.resize(faces_num*3);

	stretch_plastic_strain.resize(faces_num*3);
	stretch_plastic_strain_hardening.resize(faces_num*3);

	stretch_stick_t.resize(faces_num*3);
	stretch_anchor_strain.resize(faces_num*3);
	stretch_strain_prev.resize(faces_num*3);

	for (int f = 0; f < faces_num; ++f) {

		Face* face = mesh.faces[f];

		for (int i = 0; i < 3; ++i) {
			stretch_yield_t.at(f * 3 + i) = &face->friction_plastic_states.at(i).yield_t;
			stretch_yield_strain.at(f * 3 + i) = &face->friction_plastic_states.at(i).yield_strain;
			stretch_plastic_direction.at(f * 3 + i) = &face->friction_plastic_states.at(i).plastic_direction;

			stretch_plastic_strain.at(f * 3 + i) = &face->friction_plastic_states.at(i).plastic_strain;
			stretch_plastic_strain_hardening.at(f * 3 + i) = &face->friction_plastic_states.at(i).plastic_strain_hardening;

			stretch_stick_t.at(f * 3 + i) = &face->friction_plastic_states.at(i).stick_t;
			stretch_anchor_strain.at(f * 3 + i) = &face->friction_plastic_states.at(i).anchor_strain;
			stretch_strain_prev.at(f * 3 + i) = &face->friction_plastic_states.at(i).strain_prev;
		}
	}
}

size_t MeshLog::cal_element_num() const {
	return nodes_num * node_info_num + edges_num * edge_info_num + faces_num * face_info_num;
}

void MeshLog::save_mesh_binary(std::filesystem::path filename) const
{
	if (exists(filename)) {
		std::cerr << "File exists. Delete it to avoid overwriting exiting file." << std::endl;
		exit(1);
	}
	
	std::vector<double> all_data(cal_element_num());

	// Nodes
	for (size_t n = 0; n < nodes_num; ++n) {
		for (unsigned int i = 0; i < 3; ++i) {
			all_data.at(n * 3 + i) = (*(pos.at(n)))(i);
			all_data.at(nodes_num * 3 + n * 3 + i) = (*(pos_0.at(n)))(i);
			all_data.at(nodes_num * 3 * 2 + n * 3 + i) = (*(vel.at(n)))(i);
			all_data.at(nodes_num * 3 * 3 + n * 3 + i) = (*(acceleration.at(n)))(i);
		}
	}

	// Edges
	size_t start_idx{ nodes_num * node_info_num };

	for (size_t e = 0; e < edges_num; ++e) {
		all_data.at(start_idx + e) = *bend_yield_t.at(e);
		all_data.at(start_idx + edges_num + e) = *bend_yield_strain.at(e);
		all_data.at(start_idx + edges_num * 2 + e) = *bend_plastic_direction.at(e);

		all_data.at(start_idx + edges_num * 3 + e) = *bend_plastic_strain.at(e);
		all_data.at(start_idx + edges_num * 4 + e) = *bend_plastic_strain_hardening.at(e);

		all_data.at(start_idx + edges_num * 5 + e) = *bend_stick_t.at(e);
		all_data.at(start_idx + edges_num * 6 + e) = *bend_anchor_strain.at(e);
		all_data.at(start_idx + edges_num * 7 + e) = *bend_strain_prev.at(e);
	}

	// Faces
	start_idx += (edges_num * edge_info_num);

	for (size_t f = 0; f < faces_num; ++f) {
		for (unsigned int i = 0; i < 3; ++i) {
			all_data.at(start_idx + f * 3 + i) = *(stretch_yield_t.at(f * 3 + i));
			all_data.at(start_idx + faces_num * 3 + f * 3 + i) = *stretch_yield_strain.at(f * 3 + i);
			all_data.at(start_idx + faces_num * 3 * 2 + f * 3 + i) = *stretch_plastic_direction.at(f * 3 + i);

			all_data.at(start_idx + faces_num * 3 * 3 + f * 3 + i) = *stretch_plastic_strain.at(f * 3 + i);
			all_data.at(start_idx + faces_num * 3 * 4 + f * 3 + i) = *stretch_plastic_strain_hardening.at(f * 3 + i);

			all_data.at(start_idx + faces_num * 3 * 5 + f * 3 + i) = *stretch_stick_t.at(f * 3 + i);
			all_data.at(start_idx + faces_num * 3 * 6 + f * 3 + i) = *stretch_anchor_strain.at(f * 3 + i);
			all_data.at(start_idx + faces_num * 3 * 7 + f * 3 + i) = *stretch_strain_prev.at(f * 3 + i);
		}
	}

	std::ofstream file_save {filename, std::ios::out | std::ios::binary};

	file_save.write(reinterpret_cast<char*>(all_data.data()), all_data.size() * sizeof(double));

	file_save.close();
}

void MeshLog::load_mesh_binary(std::filesystem::path filename)
{
	
	if (!exists(filename)) {
		std::cerr << "Trying to read unexist file. Confirm file path is correct." << std::endl;
		exit(1);
	}

	std::vector<double> all_data(cal_element_num());

	std::ifstream file_read {filename, std::ios::in | std::ios::binary};

	file_read.read(reinterpret_cast<char*>(all_data.data()), all_data.size() * sizeof(double));

	file_read.close();

	// Nodes
	for (size_t n = 0; n < nodes_num; ++n) {
		for (unsigned int i = 0; i < 3; ++i) {
			(*(pos.at(n)))(i) = all_data.at(n * 3 + i);
			(*(pos_0.at(n)))(i) = all_data.at(nodes_num * 3 + n * 3 + i);
			(*(vel.at(n)))(i) = all_data.at(nodes_num * 3 * 2 + n * 3 + i);
			(*(acceleration.at(n)))(i) = all_data.at(nodes_num * 3 * 3 + n * 3 + i);
		}
	}

	// Edges
	size_t start_idx{ nodes_num * node_info_num };

	for (size_t e = 0; e < edges_num; ++e) {
		*bend_yield_t.at(e) = all_data.at(start_idx + e);
		*bend_yield_strain.at(e) = all_data.at(start_idx + edges_num + e);
		*bend_plastic_direction.at(e) = all_data.at(start_idx + edges_num * 2 + e);

		*bend_plastic_strain.at(e) = all_data.at(start_idx + edges_num * 3 + e);
		*bend_plastic_strain_hardening.at(e) = all_data.at(start_idx + edges_num * 4 + e);

		*bend_stick_t.at(e) = all_data.at(start_idx + edges_num * 5 + e);
		*bend_anchor_strain.at(e) = all_data.at(start_idx + edges_num * 6 + e);
		*bend_strain_prev.at(e) = all_data.at(start_idx + edges_num * 7 + e);
	}


	// Faces
	start_idx += (edges_num * edge_info_num);

	for (size_t f = 0; f < faces_num; ++f) {
		for (unsigned int i = 0; i < 3; ++i) {
			*stretch_yield_t.at(f * 3 + 1) = all_data.at(start_idx + f * 3 + i);
			*stretch_yield_strain.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 + f * 3 + i);
			*stretch_plastic_direction.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 * 2 + f * 3 + i);

			*stretch_plastic_strain.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 * 3 + f * 3 + i);
			*stretch_plastic_strain_hardening.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 * 4 + f * 3 + i);

			*stretch_stick_t.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 * 5 + f * 3 + i);
			*stretch_anchor_strain.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 * 6 + f * 3 + i);
			*stretch_strain_prev.at(f * 3 + 1) = all_data.at(start_idx + faces_num * 3 * 7 + f * 3 + i);
		}
	}
}

void save_obs_binary(const std::filesystem::path& filename, Mesh& mesh)
{
	if (exists(filename)) {
		std::cerr << "File exists. Delete it to avoid overwriting exiting file." << std::endl;
		exit(1);
	}

	std::vector<double> vert_pos(mesh.nodes.size() * 3);

	for (size_t n = 0; n < mesh.nodes.size(); ++n) {
		for (unsigned int i = 0; i < 3; ++i) {
			vert_pos.at(n * 3 + i) = mesh.nodes.at(n)->x(i);
		}
	}

	std::ofstream file_save {filename, std::ios::out | std::ios::binary};

	file_save.write(reinterpret_cast<char*>(vert_pos.data()), vert_pos.size() * sizeof(double));

	file_save.close();

}

void load_obs_binary(const std::filesystem::path& filename, Mesh& mesh)
{
	if (!exists(filename)) {
		std::cerr << "Trying to read unexist file. Confirm file path is correct." << std::endl;
		exit(1);
	}

	std::vector<double> vert_pos(mesh.nodes.size() * 3);

	std::ifstream file_read {filename, std::ios::in | std::ios::binary};

	file_read.read(reinterpret_cast<char*>(vert_pos.data()), vert_pos.size() * sizeof(double));

	file_read.close();

	for (size_t n = 0; n < mesh.nodes.size(); ++n) {
		for (unsigned int i = 0; i < 3; ++i) {
			mesh.nodes.at(n)->x(i) = vert_pos.at(n * 3 + i);
		}
	}
}

void motion_mesh_to_binary()
{
	std::filesystem::path obs_path{R"(D:\Research_3\build\motion\motion_lift_leg_siggraph\meshes_new\body_1.obj)"};

	std::filesystem::path save_binary_path{R"(D:\Research_3\build\motion\motion_lift_leg_siggraph\mesh_binary\body_1.dat)"};

	for (unsigned int m = 1; m < 1001; ++m) {

		std::cout << "Process Mesh: " << m << std::endl;

		Mesh mesh;

		obs_path.replace_filename(std::format("body_{}.obj", m));

		read_obj(obs_path, mesh);

		save_binary_path.replace_filename(std::format("body_{}.dat", m));

		save_obs_binary(save_binary_path, mesh);
	}
}

void motion_binary_to_mesh()
{	
	std::filesystem::path init_obs_path{R"(D:\Research_3\build\motion\motion_lift_leg_siggraph\meshes_new\body_1.obj)"};

	Mesh mesh;

	read_obj(init_obs_path, mesh);

	std::filesystem::path read_binary_path{R"(D:\Research_3\build\motion\motion_lift_leg_siggraph\mesh_binary\body_1.dat)"};

	std::filesystem::path save_mesh_path{R"(D:\Research_3\build\motion\motion_lift_leg_siggraph\saved_meshes\body_1.obj)"};

	for (unsigned int m = 1; m < 1001; ++m) {

		std::cout << "Process Mesh: " << m << std::endl;

		read_binary_path.replace_filename(std::format("body_{}.dat", m));

		load_obs_binary(read_binary_path, mesh);

		save_mesh_path.replace_filename(std::format("body_{}.obj", m));

		save_obj(save_mesh_path, mesh);
	}
}
