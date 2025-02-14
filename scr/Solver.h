#pragma once

#include "Eigen/Sparse"

void cpu_cg(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, double* x_ptr);

void cpu_pcg(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, double* x_ptr);

void gpu_cg(const int& dim, const int* row_offset_ptr, const int* col_index_ptr, const double* value_ptr, double* x_ptr, const double* b_prt, const bool is_profile=false);

void gpu_pcg(const int& dim, const int* row_offset_ptr, const int* col_index_ptr, const double* value_ptr, 
	const int* pre_row_offset_ptr, const int* pre_col_index_ptr, const double* pre_value_ptr, 
	double* x_ptr, const double* b_prt, const bool is_profile = false);

void cuda_cg(const int& dim, const int* offset_prt, const int* row_pointer, const double* value_ptr, double* x_ptr, const double* b_prt);