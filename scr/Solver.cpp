#include "Solver.h"

#include <iostream>

#include "cuda_runtime.h"
#include "cusparse.h"
#include "cublas_v2.h"

#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

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

typedef struct VecStruct {
	cusparseDnVecDescr_t vec;
	double* ptr;
} Vec;


int gpu_CG(cublasHandle_t       cublasHandle,
	cusparseHandle_t     cusparseHandle,
	int                  m,
	cusparseSpMatDescr_t matA,
	cusparseSpMatDescr_t matL,
	Vec                  d_B,
	Vec                  d_X,
	Vec                  d_R,
	Vec                  d_R_aux,
	Vec                  d_P,
	Vec                  d_T,
	Vec                  d_tmp,
	void* d_bufferMV,
	int                  maxIterations,
	double               tolerance) {
	const double zero = 0.0;
	const double one = 1.0;
	const double minus_one = -1.0;
	//--------------------------------------------------------------------------
	// ### 1 ### R0 = b - A * X0 (using initial guess in X)
	//    (a) copy b in R0
	cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice);
	//    (b) compute R = -A * X0 + R
	cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&minus_one, matA, d_X.vec, &one, d_R.vec,
		CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
		d_bufferMV);
	//--------------------------------------------------------------------------
	// ### 2 ### R_i_aux = L^-1 L^-T R_i
	size_t              bufferSizeL, bufferSizeLT;
	void* d_bufferL, * d_bufferLT;
	cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
	//    (a) L^-1 tmp => R_i_aux    (triangular solver)
	cusparseSpSV_createDescr(&spsvDescrL);
	cusparseSpSV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL);
	cudaMalloc(&d_bufferL, bufferSizeL);
	cusparseSpSV_analysis(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL);
	cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double));
	cusparseSpSV_solve(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);

	//    (b) L^-T R_i => tmp    (triangular solver)
	cusparseSpSV_createDescr(&spsvDescrLT);
	cusparseSpSV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
		&one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT);
	cudaMalloc(&d_bufferLT, bufferSizeLT);
	cusparseSpSV_analysis(
		cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
		&one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, d_bufferLT);
	cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double));
	cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
		&one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT);
	//--------------------------------------------------------------------------
	// ### 3 ### P0 = R0_aux
	cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice);
	//--------------------------------------------------------------------------
	// nrm_R0 = ||R||
	double nrm_R;
	cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R);
	double threshold = tolerance * nrm_R;
	//printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
	//--------------------------------------------------------------------------
	double delta;
	cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta);
	//--------------------------------------------------------------------------
	// ### 4 ### repeat until convergence based on max iterations and
	//           and relative residual
	int cnt_itr = 0;
	for (int i = 0; i < maxIterations; i++) {
		cnt_itr = i;
		//printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
		//----------------------------------------------------------------------
		// ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
		//     (a) T  = A * P_i
		cusparseSpMV(cusparseHandle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
			matA, d_P.vec, &zero, d_T.vec,
			CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
			d_bufferMV);
		//     (b) denominator = (T, P_i)
		double denominator;
		cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1, &denominator);
		//     (c) alpha = delta / denominator
		double alpha = delta / denominator;
		PRINT_INFO(delta)
			PRINT_INFO(denominator)
			PRINT_INFO(alpha)
			//----------------------------------------------------------------------
			// ### 6 ###  X_i+1 = X_i + alpha * P
			//    (a) X_i+1 = -alpha * T + X_i
			cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1, d_X.ptr, 1);
		//----------------------------------------------------------------------
		// ### 7 ###  R_i+1 = R_i - alpha * (A * P)
		//    (a) R_i+1 = -alpha * T + R_i
		double minus_alpha = -alpha;
		cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1, d_R.ptr, 1);
		//----------------------------------------------------------------------
		// ### 8 ###  check ||R_i+1|| < threshold
		cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R);
		PRINT_INFO(nrm_R)
			if (nrm_R < threshold)
				break;
		//----------------------------------------------------------------------
		// ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
		//    (a) L^-1 R_i+1 => tmp    (triangular solver)
		cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double));
		cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double));
		cusparseSpSV_solve(cusparseHandle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&one, matL, d_R.vec, d_tmp.vec,
			CUDA_R_64F,
			CUSPARSE_SPSV_ALG_DEFAULT,
			spsvDescrL);
		//    (b) L^-T tmp => R_aux_i+1    (triangular solver)
		cusparseSpSV_solve(cusparseHandle,
			CUSPARSE_OPERATION_TRANSPOSE,
			&one, matL, d_tmp.vec,
			d_R_aux.vec, CUDA_R_64F,
			CUSPARSE_SPSV_ALG_DEFAULT,
			spsvDescrLT);
		//----------------------------------------------------------------------
		// ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
		//    (a) delta_new => (R_i+1, R_aux_i+1)
		double delta_new;
		cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta_new);
		//    (b) beta => delta_new / delta
		double beta = delta_new / delta;
		PRINT_INFO(delta_new)
			PRINT_INFO(beta)
			delta = delta_new;
		//----------------------------------------------------------------------
		// ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
		//    (a) P = beta * P
		cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1);
		//    (b) P = R_aux + P
		cublasDaxpy(cublasHandle, m, &one, d_R_aux.ptr, 1, d_P.ptr, 1);
	}
	//--------------------------------------------------------------------------
	printf("Check Solution\n"); // ||R = b - A * X||
	//    (a) copy b in R
	cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice);
	// R = -A * X + R
	cusparseSpMV(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
		matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV);
	// check ||R||
	cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R);
	printf("Final error norm = %e\n", nrm_R);
	printf("Iterations:%d\n", cnt_itr);
	//--------------------------------------------------------------------------
	cusparseSpSV_destroyDescr(spsvDescrL);
	cusparseSpSV_destroyDescr(spsvDescrLT);
	cudaFree(d_bufferL);
	cudaFree(d_bufferLT);
	return EXIT_SUCCESS;
}

void cuda_cg(const int& dim, const int* row_offset_ptr, const int* col_index_ptr, const double* value_ptr, double* x_ptr, const double* b_ptr)
{
	
	const int maxIterations = 10000;
	const double tolerance = 1e-8f;

	int num_offsets{ dim + 1 };
	int nnz{ row_offset_ptr[dim] };
	
	int* d_A_rows, * d_A_columns;
	double* d_A_values, * d_L_values;
	Vec d_B, d_X, d_R, d_R_aux, d_P, d_T, d_tmp;

	cudaMalloc((void**)&d_A_rows, num_offsets * sizeof(int));
	cudaMalloc((void**)&d_A_columns, nnz * sizeof(int));
	cudaMalloc((void**)&d_A_values, nnz * sizeof(double));
	cudaMalloc((void**)&d_L_values, nnz * sizeof(double));

	cudaMalloc((void**)&d_B.ptr, dim * sizeof(double));
	cudaMalloc((void**)&d_X.ptr, dim * sizeof(double));
	cudaMalloc((void**)&d_R.ptr, dim * sizeof(double));
	cudaMalloc((void**)&d_R_aux.ptr, dim * sizeof(double));
	cudaMalloc((void**)&d_P.ptr, dim * sizeof(double));
	cudaMalloc((void**)&d_T.ptr, dim * sizeof(double));
	cudaMalloc((void**)&d_tmp.ptr, dim * sizeof(double));

	cudaMemcpy(d_A_rows, row_offset_ptr, num_offsets * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_columns, col_index_ptr, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_values, value_ptr, nnz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L_values, value_ptr, nnz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_X.ptr, x_ptr, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.ptr, b_ptr, dim * sizeof(double), cudaMemcpyHostToDevice);

	cublasHandle_t cublasHandle = NULL;
	cusparseHandle_t cusparseHandle = NULL;

	cublasCreate(&cublasHandle);
	cusparseCreate(&cusparseHandle);

	cusparseCreateDnVec(&d_B.vec, dim, d_B.ptr, CUDA_R_64F);
	cusparseCreateDnVec(&d_X.vec, dim, d_X.ptr, CUDA_R_64F);
	cusparseCreateDnVec(&d_R.vec, dim, d_R.ptr, CUDA_R_64F);
	cusparseCreateDnVec(&d_R_aux.vec, dim, d_R_aux.ptr, CUDA_R_64F);

	cusparseCreateDnVec(&d_P.vec, dim, d_P.ptr, CUDA_R_64F);
	cusparseCreateDnVec(&d_T.vec, dim, d_T.ptr, CUDA_R_64F);
	cusparseCreateDnVec(&d_tmp.vec, dim, d_tmp.ptr, CUDA_R_64F);

	cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
	cusparseSpMatDescr_t matA, matL;
	int* d_L_rows = d_A_rows;
	int* d_L_columns = d_A_columns;
	cusparseFillMode_t   fill_lower = CUSPARSE_FILL_MODE_LOWER;
	cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

	// A
	cusparseCreateCsr(&matA, dim, dim, nnz, d_A_rows, d_A_columns, d_A_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);

	// L
	cusparseCreateCsr(&matL, dim, dim, nnz, d_L_rows, d_L_columns, d_L_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, baseIdx, CUDA_R_64F);

	cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));

	cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit));

	// ### Preparation
	const double alpha = 1.0;
	size_t bufferSizeMV;
	void* d_bufferMV;
	double beta = 0.0;

	cusparseSpMV_bufferSize( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV); 

	cudaMalloc(&d_bufferMV, bufferSizeMV);

	// X0 = 0
	cudaMemset(d_X.ptr, 0x0, dim * sizeof(double));

	//--------------------------------------------------------------------------
	// Perform Incomplete-Cholesky factorization of A (csric0) -> L, L^T
	cusparseMatDescr_t descrM;
	csric02Info_t      infoM = NULL;
	int                bufferSizeIC = 0;
	void* d_bufferIC;

	cusparseCreateMatDescr(&descrM);
	cusparseSetMatIndexBase(descrM, baseIdx);
	cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseCreateCsric02Info(&infoM);

	cusparseDcsric02_bufferSize(
		cusparseHandle, dim, nnz, descrM, d_L_values,
		d_A_rows, d_A_columns, infoM, &bufferSizeIC);
	cudaMalloc(&d_bufferIC, bufferSizeIC);
	cusparseDcsric02_analysis(
		cusparseHandle, dim, nnz, descrM, d_L_values,
		d_A_rows, d_A_columns, infoM,
		CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC);
	int structural_zero;
	cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
		&structural_zero);

	// M = L * L^T
	cusparseDcsric02(
		cusparseHandle, dim, nnz, descrM, d_L_values,
		d_A_rows, d_A_columns, infoM,
		CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC);

	// Find numerical zero
	int numerical_zero;
	cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
		&numerical_zero);

	cusparseDestroyCsric02Info(infoM);
	cusparseDestroyMatDescr(descrM);
	cudaFree(d_bufferIC);

	// ### Run CG computation ###
	//printf("CG loop:\n");
	gpu_CG(cublasHandle, cusparseHandle, dim, 
		matA, matL, d_B, d_X, d_R, d_R_aux, d_P, d_T,
		d_tmp, d_bufferMV, maxIterations, tolerance);

	cudaMemcpy(x_ptr, d_X.ptr, dim * sizeof(double), cudaMemcpyDeviceToHost);

	cusparseDestroyDnVec(d_B.vec);
	cusparseDestroyDnVec(d_X.vec);
	cusparseDestroyDnVec(d_R.vec);
	cusparseDestroyDnVec(d_R_aux.vec);
	cusparseDestroyDnVec(d_P.vec);
	cusparseDestroyDnVec(d_T.vec);
	cusparseDestroyDnVec(d_tmp.vec);
	cusparseDestroySpMat(matA);
	cusparseDestroySpMat(matL);
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	//free(h_A_rows);
	//free(h_A_columns);
	//free(h_A_values);
	//free(h_X);

	cudaFree(d_X.ptr);
	cudaFree(d_B.ptr);
	cudaFree(d_R.ptr);
	cudaFree(d_R_aux.ptr);
	cudaFree(d_P.ptr);
	cudaFree(d_T.ptr);
	cudaFree(d_tmp.ptr);
	cudaFree(d_A_values);
	cudaFree(d_A_columns);
	cudaFree(d_A_rows);
	cudaFree(d_L_values);
	cudaFree(d_bufferMV);
}

void cpu_cg(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, double* x_ptr)
{
	Eigen::Map<Eigen::VectorXd> x(x_ptr, b.rows());
	x.setZero();
	
	Eigen::VectorXd r = b - A * x;
	Eigen::VectorXd d = r;

	double delta_new = r.dot(r);
	double delta_0 = delta_new;

	unsigned int iter{ 0 };
	const unsigned int max_iter{ 1000000 };
	const double eps{ 1e-9 };

	while (true) {

		if (iter > max_iter || (delta_new < std::pow(eps, 2) * delta_0)) {
			break;
		}

		Eigen::VectorXd q = A * d;
		double alpha = delta_new / (d.dot(q));

		x += alpha * d;
		r -= alpha * q;

		double delta_prev = delta_new;
		delta_new = r.dot(r);
		double beta = delta_new / delta_prev;
		d = r + beta * d;

		iter++;
	}

	std::cout << "Iter: " << iter << std::endl;
	std::cout << "Average Residual: " << std::sqrt(delta_new) / (b.rows()) << std::endl;
}

void cpu_pcg(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, double* x_ptr)
{
	std::cout << "CPU PCG" << std::endl;
	
	Eigen::Map<Eigen::VectorXd> x(x_ptr, b.rows());
	x.setZero();

	Eigen::MatrixXd M_inv = A.diagonal().asDiagonal().inverse();

	Eigen::VectorXd r = b - A * x;
	Eigen::VectorXd d = M_inv * r;
	Eigen::VectorXd s = d;

	double delta_new = r.dot(d);
	double delta_0 = delta_new;

	unsigned int iter{ 0 };
	const unsigned int max_iter{ 1000000 };
	const double eps{ 1e-9 };

	while (true) {

		if (iter > max_iter || (delta_new < std::pow(eps, 2) * delta_0)) {
			break;
		}

		Eigen::VectorXd q = A * d;
		double alpha = delta_new / (d.dot(q));

		x += alpha * d;
		r -= alpha * q;

		s = M_inv * r;
		double delta_prev = delta_new;
		delta_new = r.dot(s);
		double beta = delta_new / delta_prev;
		d = s + beta * d;

		iter += 1;
	}

	std::cout << "Iter: " << iter << std::endl;
	std::cout << "Average Residual: " << std::sqrt(delta_new) / (b.rows()) << std::endl;
}

void gpu_cg(const int& dim, const int* row_offset_ptr, const int* col_index_ptr, const double* value_ptr, double* x_ptr, const double* b_prt, const bool is_profile)
{
	int num_offsets = dim + 1;
	int nnz = row_offset_ptr[dim];

	int* dA_csrOffsets, * dA_columns;
	double* dA_values, * dX, * dY;
	double* dB;

	double alpha_one = 1.0, beta = 0.0;

	CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, num_offsets * sizeof(int)))
	CHECK_CUDA(cudaMalloc((void**)&dA_columns, nnz * sizeof(int)))
	CHECK_CUDA(cudaMalloc((void**)&dA_values, nnz * sizeof(double)))
	CHECK_CUDA(cudaMalloc((void**)&dX, dim * sizeof(double)))
	CHECK_CUDA(cudaMalloc((void**)&dB, dim * sizeof(double)))
	CHECK_CUDA(cudaMalloc((void**)&dY, dim * sizeof(double)))

	CHECK_CUDA(cudaMemcpy(dA_csrOffsets, row_offset_ptr, num_offsets * sizeof(int), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dA_columns, col_index_ptr, nnz * sizeof(int), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dA_values, value_ptr, nnz * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dX, x_ptr, dim * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dB, b_prt, dim * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemset(dY, 0.0, dim))

	// compute A * x
	cusparseHandle_t handle_mv = nullptr;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void* dBuffer_mv = nullptr;
	size_t buffersize = 0;

	CHECK_CUSPARSE(cusparseCreate(&handle_mv))
	CHECK_CUSPARSE(cusparseCreateCsr(&matA, dim, dim, nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, dim, dX, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, dim, dY, CUDA_R_64F))

	CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize))

	CHECK_CUDA(cudaMalloc(&dBuffer_mv, buffersize))

	CHECK_CUSPARSE(cusparseSpMV(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_mv))

	// computer r = b - Ax
	cublasHandle_t cublas_h = nullptr;
	double alpha_r = -1.0;
	const int incx = 1;
	const int incy = 1;

	CHECK_CUBLAS(cublasCreate(&cublas_h))

	CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha_r, dY, incx, dB, incy))

	// d = r
	double* dD;

	CHECK_CUDA(cudaMalloc((void**)&dD, dim * sizeof(double)))

	CHECK_CUDA(cudaMemcpy(dD, dB, dim * sizeof(double), cudaMemcpyDeviceToDevice))

	cusparseDnVecDescr_t vecD;

	CHECK_CUSPARSE(cusparseCreateDnVec(&vecD, dim, dD, CUDA_R_64F))

	// computer r_t * r
	double delta_new = 0.0;

	CHECK_CUBLAS(cublasDdot(cublas_h, dim, dB, incy, dB, incy, &delta_new))

	double delta_0 = delta_new;

	// Optimization Loop
	double* dQ;

	CHECK_CUDA(cudaMalloc((void**)&dQ, dim * sizeof(double)))

	cudaMemset(dQ, 0.0, dim);

	cusparseDnVecDescr_t vecQ;

	CHECK_CUSPARSE(cusparseCreateDnVec(&vecQ, dim, dQ, CUDA_R_64F))

	int iter{ 0 };

	const unsigned int max_iter{ 1000000 };
	const double eps{ 1e-9 };

	while (true) {

		if (iter > max_iter || (delta_new < std::pow(eps, 2) * delta_0)) {
			break;
		}

		CHECK_CUDA(cudaMemset(dQ, 0.0, dim))

		// compute q = A * d;
		CHECK_CUSPARSE(cusparseSpMV(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matA, vecD, &beta, vecQ, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_mv))

		// compute alpha = delta_new / (d.t() * q)
		double d_dot_q = 0.0;

		CHECK_CUBLAS(cublasDdot(cublas_h, dim, dD, incy, dQ, incy, &d_dot_q))

		//std::cout << "d_dot_q: " << d_dot_q << std::endl;

		double alpha = delta_new / d_dot_q;

		//std::cout << "alpha: " << alpha << std::endl;

		// compute x = x + alpha * d
		CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha, dD, incx, dX, incy))

		double alpha_minus = -alpha;

		// compute r = r - alpha * q
		CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha_minus, dQ, incx, dB, incy))

		// delta_prev = delta_new
		double delta_prev = delta_new;

		// delta_new = r.t() * r;
		CHECK_CUBLAS(cublasDdot(cublas_h, dim, dB, incy, dB, incy, &delta_new))

		//std::cout << "delta_new: " << delta_new << std::endl;

		double beta = delta_new / delta_prev;

		//std::cout << "beta: " << beta << std::endl;

		// d = r + beta * d

		CHECK_CUBLAS(cublasDscal(cublas_h, dim, &beta, dD, incx))
		
		CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha_one, dB, incx, dD, incy))

		iter++;

		//std::cout << "Iter: " << iter << std::endl;
		//std::cout << "Average Residual: " << std::sqrt(delta_new) / (dim) << std::endl;
	}

	if (is_profile) {
		std::cout << "Iter: " << iter << std::endl;
		std::cout << "Average Residual: " << std::sqrt(delta_new) / (dim) << std::endl;
	}
	
	// copy the solved result back
	CHECK_CUDA(cudaMemcpy(x_ptr, dX, dim * sizeof(double), cudaMemcpyDeviceToHost))

	CHECK_CUSPARSE(cusparseDestroySpMat(matA))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecQ))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecD))

	CHECK_CUSPARSE(cusparseDestroy(handle_mv))
	CHECK_CUBLAS(cublasDestroy(cublas_h))
	
	// Free memory
	CHECK_CUDA(cudaFree(dBuffer_mv))

	CHECK_CUDA(cudaFree(dA_csrOffsets))
	CHECK_CUDA(cudaFree(dA_columns))
	CHECK_CUDA(cudaFree(dA_values))
	CHECK_CUDA(cudaFree(dX))
	CHECK_CUDA(cudaFree(dY))		
	CHECK_CUDA(cudaFree(dB))
	CHECK_CUDA(cudaFree(dD))
	CHECK_CUDA(cudaFree(dQ))

	// Reset CUDA
	CHECK_CUDA(cudaDeviceReset())
}

void gpu_pcg(const int& dim, const int* row_offset_ptr, const int* col_index_ptr, const double* value_ptr, 
	const int* pre_row_offset_ptr, const int* pre_col_index_ptr, const double* pre_value_ptr, 
	double* x_ptr, const double* b_prt,  const bool is_profile)
{
	std::vector<double> zeros(dim, 0.0);
	
	int num_offsets = dim + 1;
	int nnz = row_offset_ptr[dim];
	int m_nnz = pre_row_offset_ptr[dim];

	int* dA_csrOffsets, * dA_columns;
	int* dM_csrOffsets, * dM_columns;
	double* dA_values, *dM_values, * dX, * dY;
	double* dB;

	double alpha_one = 1.0, beta = 0.0;

	CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, num_offsets * sizeof(int)))
	CHECK_CUDA(cudaMalloc((void**)&dA_columns, nnz * sizeof(int)))
	CHECK_CUDA(cudaMalloc((void**)&dA_values, nnz * sizeof(double)))

	CHECK_CUDA(cudaMalloc((void**)&dM_csrOffsets, num_offsets * sizeof(int)))
	CHECK_CUDA(cudaMalloc((void**)&dM_columns, m_nnz * sizeof(int)))
	CHECK_CUDA(cudaMalloc((void**)&dM_values, m_nnz * sizeof(double)))

	CHECK_CUDA(cudaMalloc((void**)&dX, dim * sizeof(double)))
	CHECK_CUDA(cudaMalloc((void**)&dB, dim * sizeof(double)))
	CHECK_CUDA(cudaMalloc((void**)&dY, dim * sizeof(double)))
	
	// Copy Matrix A Host to Device
	CHECK_CUDA(cudaMemcpy(dA_csrOffsets, row_offset_ptr, num_offsets * sizeof(int), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dA_columns, col_index_ptr, nnz * sizeof(int), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dA_values, value_ptr, nnz * sizeof(double), cudaMemcpyHostToDevice))
	// Copy Precondiction Matrix Host to Device
	CHECK_CUDA(cudaMemcpy(dM_csrOffsets, pre_row_offset_ptr, num_offsets * sizeof(int), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dM_columns, pre_col_index_ptr, m_nnz * sizeof(int), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dM_values, pre_value_ptr, m_nnz * sizeof(double), cudaMemcpyHostToDevice))

	CHECK_CUDA(cudaMemcpy(dX, x_ptr, dim * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dB, b_prt, dim * sizeof(double), cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpy(dY, zeros.data(), dim * sizeof(double), cudaMemcpyHostToDevice))

	// compute A * x
	cusparseHandle_t handle_mv = nullptr;
	cusparseSpMatDescr_t matA, matMInv;
	cusparseDnVecDescr_t vecX, vecY;
	void* dBuffer_mv = nullptr;
	size_t buffersize = 0;

	CHECK_CUSPARSE(cusparseCreate(&handle_mv))
	CHECK_CUSPARSE(cusparseCreateCsr(&matA, dim, dim, nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, dim, dX, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, dim, dY, CUDA_R_64F))

	CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize))

	CHECK_CUDA(cudaMalloc(&dBuffer_mv, buffersize))

	CHECK_CUSPARSE(cusparseSpMV(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_mv))

	// computer r = b - Ax
	cublasHandle_t cublas_h = nullptr;
	double alpha_r = -1.0;
	const int incx = 1;
	const int incy = 1;

	CHECK_CUBLAS(cublasCreate(&cublas_h))

	CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha_r, dY, incx, dB, incy))

	// d = M^(-1) * r
	double* dD;

	cusparseDnVecDescr_t vecD, vecR;

	CHECK_CUDA(cudaMalloc((void**)&dD, dim * sizeof(double)))

	CHECK_CUDA(cudaMemcpy(dD, zeros.data(), dim * sizeof(double), cudaMemcpyHostToDevice))

	CHECK_CUSPARSE(cusparseCreateDnVec(&vecD, dim, dD, CUDA_R_64F))

	CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, dim, dB, CUDA_R_64F))

	CHECK_CUSPARSE(cusparseCreateCsr(&matMInv, dim, dim, m_nnz, dM_csrOffsets, dM_columns, dM_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))	/*Create Preconditional Sparse Matrix*/

	CHECK_CUSPARSE(cusparseSpMV(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matMInv, vecR, &beta, vecD, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_mv))

	// computer r_t * r
	double delta_new = 0.0;

	CHECK_CUBLAS(cublasDdot(cublas_h, dim, dB, incy, dB, incy, &delta_new))

	double delta_0 = delta_new;

	// Optimization Loop
	double* dQ, *dS;

	cusparseDnVecDescr_t vecQ, vecS;

	CHECK_CUDA(cudaMalloc((void**)&dQ, dim * sizeof(double)))

	CHECK_CUDA(cudaMemcpy(dQ, zeros.data(), dim * sizeof(double), cudaMemcpyHostToDevice))

	CHECK_CUDA(cudaMalloc((void**)&dS, dim * sizeof(double)))

	CHECK_CUDA(cudaMemcpy(dS, zeros.data(), dim * sizeof(double), cudaMemcpyHostToDevice))

	CHECK_CUSPARSE(cusparseCreateDnVec(&vecQ, dim, dQ, CUDA_R_64F))

	CHECK_CUSPARSE(cusparseCreateDnVec(&vecS, dim, dS, CUDA_R_64F))

	int iter{ 0 };

	const unsigned int max_iter{ 1000000 };
	const double eps{ 1e-9 };

	while (true) {

		if (iter > max_iter || (delta_new < std::pow(eps, 2) * delta_0)) {
			break;
		}

		// compute q = A * d;
		CHECK_CUSPARSE(cusparseSpMV(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matA, vecD, &beta, vecQ, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_mv))

		// compute alpha = delta_new / (d.t() * q)
		double d_dot_q = 0.0;

		CHECK_CUBLAS(cublasDdot(cublas_h, dim, dD, incy, dQ, incy, &d_dot_q))

		double alpha = delta_new / d_dot_q;

		// compute x = x + alpha * d
		CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha, dD, incx, dX, incy))

		double alpha_minus = -alpha;

		// compute r = r - alpha * q
		CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha_minus, dQ, incx, dB, incy))

		// computer s = M^(-1) * r

		CHECK_CUSPARSE(cusparseSpMV(handle_mv, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_one, matMInv, vecR, &beta, vecS, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_mv))

		// delta_prev = delta_new
		double delta_prev = delta_new;

		// delta_new = r.t() * s;
		CHECK_CUBLAS(cublasDdot(cublas_h, dim, dB, incy, dS, incy, &delta_new))

		double beta_cg = delta_new / delta_prev;

		// d = s + beta * d

		CHECK_CUBLAS(cublasDscal(cublas_h, dim, &beta_cg, dD, incx))

		CHECK_CUBLAS(cublasDaxpy(cublas_h, dim, &alpha_one, dS, incx, dD, incy))

		iter++;
	}

	if (is_profile) {
		std::cout << "Iter: " << iter << std::endl;
		std::cout << "Average Residual: " << std::sqrt(delta_new) / (dim) << std::endl;
	}

	// copy the solved result back
	CHECK_CUDA(cudaMemcpy(x_ptr, dX, dim * sizeof(double), cudaMemcpyDeviceToHost))

	CHECK_CUSPARSE(cusparseDestroySpMat(matA))
	CHECK_CUSPARSE(cusparseDestroySpMat(matMInv))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecR))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecD))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecS))
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecQ))

	CHECK_CUSPARSE(cusparseDestroy(handle_mv))
	CHECK_CUBLAS(cublasDestroy(cublas_h))

	// Free memory
	CHECK_CUDA(cudaFree(dBuffer_mv))

	CHECK_CUDA(cudaFree(dA_csrOffsets))
	CHECK_CUDA(cudaFree(dA_columns))
	CHECK_CUDA(cudaFree(dA_values))

	CHECK_CUDA(cudaFree(dM_csrOffsets))
	CHECK_CUDA(cudaFree(dM_columns))
	CHECK_CUDA(cudaFree(dM_values))

	CHECK_CUDA(cudaFree(dX))
	CHECK_CUDA(cudaFree(dY))
	CHECK_CUDA(cudaFree(dB))
	CHECK_CUDA(cudaFree(dD))
	CHECK_CUDA(cudaFree(dQ))
	CHECK_CUDA(cudaFree(dS))

	// Reset CUDA
	CHECK_CUDA(cudaDeviceReset())
}