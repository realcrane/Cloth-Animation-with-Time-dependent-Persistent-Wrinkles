#include "cuda_sim.cuh"

#include "Simulation.h"
#include "MeshIO.h"

#include <thread>
#include <fstream>
#include <ranges>

#include <chrono>
#include <Windows.h>
#include <functional>

#include "cuda_runtime.h"
#include "cusparse.h"
#include "cublas_v2.h"

using sparse_tri = Eigen::Triplet<double>;
using sparse_M = Eigen::SparseMatrix<double>;

using namespace std::literals::chrono_literals;

#define MY_TIMING(func) {auto start = std::chrono::high_resolution_clock::now(); \
	func;\
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();\
	std::cout << "Duration: " << duration << " ms"<< std::endl;}


#define CHECK_CUDA(func) {cudaError_t status = (func);\
	if(status != cudaSuccess){\
		printf("CUDA API failed at line %d with error: %s (%d)\n",\
		__LINE__, cudaGetErrorString(status), status);\
		exit(1);}}

#define CHECK_CUSPARSE(func) { cusparseStatus_t status = (func);\
	if (status != CUSPARSE_STATUS_SUCCESS){\
		printf("CUDA API failed at line %d with error: %s (%d)\n", \
			__LINE__, cusparseGetErrorString(status), status); \
			exit(1);}}

#define CHECK_CUBLAS(func) {cublasStatus_t err_ = (func);\
	if(err_ != CUBLAS_STATUS_SUCCESS){\
		printf("cublas error %d at %s:%d\n",err_, __FILE__, __LINE__);\
		exit(1);}}

void add(const int& a, const int& b, int* c){
	*c = a + b;
}

void sum_parallel()
{
	int a[10]{ 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int b[10]{ 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int c[10]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	std::vector<std::thread> threads(10);

	for (auto i = 0; i < 10; i++) {
		threads[i] = std::thread(add, a[i], b[i], &c[i]);
	}

	for (auto& e : threads) {
		e.join();
	}
}

void cuda_mv() {
	
	const int A_num_rows = 4;
	const int A_num_cols = 4;
	const int A_nnz = 9;

	int hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
	int hA_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
	float hA_values[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
	float hX[] = { 1.0f, 2.0f, 3.0f, 4.0f };
	float hY[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	float hY_result[] = { 19.0f, 8.0f, 51.0f, 52.0f };
	float alpha = 1.0f;
	float beta = 0.0f;

	// Device
	int* dA_csrOffsets = NULL, * dA_columns = NULL;
	float* dA_values = NULL, * dX = NULL, * dY = NULL;
	cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
	cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int));
	cudaMalloc((void**)&dA_values, A_nnz * sizeof(float));
	cudaMalloc((void**)&dX, A_num_cols * sizeof(float));
	cudaMalloc((void**)&dY, A_num_rows * sizeof(float));

	cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice);

	cusparseHandle_t handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void* dBuffer = NULL;
	size_t bufferSize = 0;

	cusparseCreate(&handle);
	cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz, dA_csrOffsets, dA_columns, dA_values,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

	cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F);
	cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F);
	
	cusparseSpMV_bufferSize( handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

	cudaMalloc(&dBuffer, bufferSize);

	cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	cusparseDestroy(handle);

	cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost);

	int correct = 1;
	for (int i = 0; i < A_num_rows; i++) {
		std::cout << "i: " << i << std::endl;
		std::cout << hY[i] << std::endl;
		if (hY[i] != hY_result[i]) { // direct floating point comparison is not
			correct = 0;             // reliable
			break;
		}
	}
	if (correct)
		printf("spmv_csr_example test PASSED\n");
	else
		printf("spmv_csr_example test FAILED: wrong result\n");

	cudaFree(dBuffer);

	cudaFree(dA_csrOffsets);
	cudaFree(dA_columns);
	cudaFree(dA_values);
	cudaFree(dX);
	cudaFree(dY);
}

void sum_number(int& a) {

	//std::cout << a << std::endl;

	std::cout << "thread Id: " << std::this_thread::get_id() << std::endl;

	for (int i = 0; i < 10; ++i) {
		a += i;
		
		//for (int j = 0; j < 10; ++j) {
		//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
		//}
	}
		
}

void seq_sum() {

	int as[3];

	for (unsigned int i = 0; i < 3; ++i) {
		sum_number(as[i]);
	}


}

void sum_paralle() {

	std::vector<std::thread> threads(10);
	std::vector<int> as(10);

	for (unsigned int i = 0; i < 10; ++i) {
		threads[i] = std::thread(sum_number, std::ref(as[i]) );
	}

	for (auto& entry : threads)
		entry.join();
}

void sum_numbers(std::vector<double>::iterator itr, int cnt, double& sum) {
	for (int i = 0; i < cnt; ++i) {
		sum += itr[i];
	}
}

void test_range_view() {

	//std::vector<double> a {1.0, 2.0, 3.0};
	//std::vector<double> b {4.0, 5.0, 6.0};

	//std::cout << "Before concat" << std::endl;

	//for (auto a_v : a) {
	//	std::cout << a_v << std::endl;
	//}

	//a.insert(a.end(), b.begin(), b.end());

	//std::cout << "After concat" << std::endl;

	//for (auto a_v : a) {
	//	std::cout << a_v << std::endl;
	//}

	//std::cout << "Joint View" << std::endl;

	//std::vector<std::vector<double>> vectors_to_join = {a, b};

	//auto a_join = std::ranges::join_view(vectors_to_join);

	//for (auto a_s : a_join)
	//	std::cout << a_s << std::endl;

	//std::cout << *a_join.begin() << std::endl;
	//std::cout << a_join.back() << std::endl;



	sparse_M spar_mat(3, 3);

	std::vector<std::vector<sparse_tri>> spar_mat_eles;

	spar_mat_eles.push_back(std::vector<sparse_tri>());
	spar_mat_eles.push_back(std::vector<sparse_tri>());

	spar_mat_eles[0].emplace_back(sparse_tri(0, 0, 1.0));

	spar_mat_eles[1].emplace_back(sparse_tri(1, 1, 2.0));
	spar_mat_eles[1].emplace_back(sparse_tri(2, 2, 3.0));

	//spar_mat_eles[0].insert(spar_mat_eles[0].end(), spar_mat_eles[1].begin(), spar_mat_eles[1].end());

	//spar_mat.setFromTriplets(spar_mat_eles[0].begin(), spar_mat_eles[0].end());

	auto spar_mat_eles_view = std::ranges::join_view(spar_mat_eles);
	
	spar_mat.setFromTriplets(spar_mat_eles_view.begin(), spar_mat_eles_view.end());

	std::cout << spar_mat << std::endl;
}


int main(void)
{	
	
	//funcCaller();

	//Eigen::initParallel();

	//Eigen::setNbThreads(20);

	Simulation sim_session{ std::filesystem::path(R"(.\SimConfig.json)") };

	//sim_session.clothes[0].mesh.mesh_info();

	//sim_session.render_mesh();

	sim_session.physics();

	//MY_TIMING(sim_session.physics())

	//gl_render();





	//motion_binary_to_mesh();


	return 0;
}
