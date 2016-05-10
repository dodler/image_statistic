/*
 ============================================================================

 Author      : Artyom Lyan
 Version     :
 Copyright   : Shareable, my bachelor degree work
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

#include <vector_types.h>
#include <ctime>

#include <iostream>
#include <numeric>
#include <cstdlib>
#include <string>
#include <map>

#include "image_statistic.h"

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * function returns size of matrix defined higher in bytes
 */
int get_matrix_size_bytes(int cols, int rows){
	return (cols*rows)*sizeof(int);
}

/**
 * this is function that returns image density at point
 * m1 is column number
 * m2 is position in row
 */
__device__ int f(int m1, int m2, int cols, int rows, int* pic){
	if (m1 < cols && m2 < rows && m1 >= 0 && m2 >= 0){
		return pic[m1*rows + m2];
	}else{
		return 0;
	}
}

/**
 * this is indicator function
 * m1,m2 are points
 * d_m1 and d_m2 are distances for 2 points
 * i and j are densities for m1,m2 and m1+d_m1, m2_d_m2 points respectively
 */
__device__ int q_ij(int m1, int m2, int d_m1, int d_m2, int i, int j, int cols, int rows, int* pic){
	int result = 0;
	if (f(m1,m2, cols, rows, pic) == i && f(m1+d_m1, m2+d_m2, cols, rows, pic) == j){
		result = 1;
	}
	return result;
}

/**
 * this function returns non-normed values of adjacency matrix
 * here i and j are intensity levels
 * this function will be called once for single row and string
 * M2 is length of string
 * function will be launched N times, N - number of rows
 * c_values stores source image in format of vector
 */
__global__ void c_dm1_dm2(int i, int j, int cols, int rows, int dm1, int dm2, int* c_values, int* res){
	int index = threadIdx.x + blockDim.x* blockIdx.x; // here index equals to string number
	res[index] = 0;


	for(int m1 = 0; m1<cols; m1++){
		res[index] += q_ij(index, m1, dm1, dm2, i,j,cols, rows, c_values);
	}
}

/**
 * result will be stored in first vector
 */
__global__ void sum_vector(double* v1, double* v2, int len, int stride){
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	for(int i = index*stride; i<(index+1)*stride; i++){
		v1[i] += v2[i];
	}
}

__global__ void devide_vector(double* v1, double devide_by, int stride){
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	for(int i = index*stride; i<(index+1)*stride; i++){
		v1[i] /= devide_by;
	}
}


/**
 * in this program i handle each string of picture in single thread
 */

double* calc_adjacency_matrix(int dm1, int dm2, int* values, int cols, int rows, int max_i, int max_j){

	int m_size = get_matrix_size_bytes(cols, rows);

	int* c_values;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_values, m_size));
	CUDA_CHECK_RETURN(cudaMemcpy(c_values, values, m_size, cudaMemcpyHostToDevice));

	int blockDim = 1, threadDim = cols;
	int* c_res;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_res, blockDim*threadDim*sizeof(int)));

	int* res = new int[blockDim*threadDim];

	int** res_mat = new int*[max_i];

	int total_pairs = 0;

	for(int i = 0; i<max_i; i++){
		res_mat[i] = new int[max_j];
		for(int j = 0; j<max_j; j++){
			c_dm1_dm2<<<blockDim, threadDim>>>(i,j,cols, rows, dm1, dm2, c_values, c_res);
			CUDA_CHECK_RETURN(cudaMemcpy(res, c_res, blockDim*threadDim*sizeof(int), cudaMemcpyDeviceToHost));

			res_mat[i][j] = std::accumulate(res, res+blockDim*threadDim, 0);
			total_pairs += res_mat[i][j];
		}
	}

	double* normalized_res_mat = new double[max_i*max_j];
	for(int i = 0; i<max_i; i++){
		for(int j = 0; j<max_j;j++){
			normalized_res_mat[i*max_i + j] = (double)res_mat[i][j]/total_pairs;
		}
	}

	delete []res;
	for(int i = 0; i<max_i; i++){
		delete []res_mat[i];
	}
	delete []res_mat;

	return normalized_res_mat;
}

double* calc_symmetric_adjacency_matrix(int* pic, int cols, int rows, int dm1, int dm2, int max_i, int max_j){

	int size = max_i * max_j;

	double* adj_matr_1 = calc_adjacency_matrix(dm1, dm2, pic, cols, rows, max_i, max_j);
	double* adj_matr_2 = calc_adjacency_matrix(-dm1, dm2, pic, cols, rows, max_i, max_j);
	double* adj_matr_3 = calc_adjacency_matrix(dm1, -dm2, pic, cols, rows, max_i, max_j);
	double* adj_matr_4 = calc_adjacency_matrix(-dm1, -dm2, pic, cols, rows, max_i, max_j);

	double *c_adj_matr1,*c_adj_matr2,*c_adj_matr3,*c_adj_matr4;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_adj_matr1, size*sizeof(double)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_adj_matr2, size*sizeof(double)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_adj_matr3, size*sizeof(double)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&c_adj_matr4, size*sizeof(double)));

	CUDA_CHECK_RETURN(cudaMemcpy(c_adj_matr1, adj_matr_1, size*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(c_adj_matr2, adj_matr_2, size*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(c_adj_matr3, adj_matr_3, size*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(c_adj_matr4, adj_matr_4, size*sizeof(double), cudaMemcpyHostToDevice));

	sum_vector<<<1, max_i>>>(c_adj_matr1, c_adj_matr2, size, max_i);
	sum_vector<<<1, max_i>>>(c_adj_matr1, c_adj_matr3, size, max_i);
	sum_vector<<<1, max_i>>>(c_adj_matr1, c_adj_matr4, size, max_i);

	devide_vector<<<1, max_i>>>(c_adj_matr1, 4.0, max_i);

	CUDA_CHECK_RETURN(cudaMemcpy(adj_matr_1, c_adj_matr1, size*sizeof(double), cudaMemcpyDeviceToHost));

	delete []adj_matr_2;
	delete []adj_matr_3;
	delete []adj_matr_4;

	CUDA_CHECK_RETURN(cudaFree(c_adj_matr1));
	CUDA_CHECK_RETURN(cudaFree(c_adj_matr2));
	CUDA_CHECK_RETURN(cudaFree(c_adj_matr3));
	CUDA_CHECK_RETURN(cudaFree(c_adj_matr4));

	return adj_matr_1;
}

void test(){
	double* v1 = new double[4];
	double* v2 = new double[4];

	for(int i = 0; i<4; i++){
		v1[i] = i;
		v2[i] = i;
	}

	print_vector(v1, 4);
	print_vector(v2, 4);

	double *c_v1, *c_v2;

	cudaMalloc((void**)&c_v1, 4*sizeof(double));
	cudaMalloc((void**)&c_v2, 4*sizeof(double));

	cudaMemcpy(c_v1, v1, 4*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(c_v2, v2, 4*sizeof(double), cudaMemcpyHostToDevice);

	sum_vector<<<1,2>>>(c_v1, c_v2, 4, 2);

	cudaMemcpy(v1, c_v1, 4*sizeof(double), cudaMemcpyDeviceToHost);
	print_vector(v1, 4);

	cudaFree(c_v1);
	cudaFree(c_v2);
	delete []v1;
	delete []v2;
}

void test2(void) {

	// here adjacency matrix calculation is started

	srand(time(NULL));

	static const int rows = 1024, cols = 1024,
			dm1 = 20, dm2 = 20, max_i = 10, max_j = 10;

	int* values = prepare_matrix(cols, rows);
	const int m_size = get_matrix_size_bytes(cols,rows);

	cout << endl;

	double* matr = calc_symmetric_adjacency_matrix(values, cols, rows, dm1, dm2, max_i, max_j);
	print_vector(matr, max_i, max_j);
	cout << "finished" << endl;

	delete []values;
	delete []matr;
}

void print_vector(double* v, int cols, int rows){
	for(int i = 0; i<cols; i++){
		for(int j = 0; j<rows; j++){
			cout << v[i*cols + j] << "|";
		}
		cout << endl;
	}
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

/**
 * this method prepares image (currently matrix)
 * to be proceed on cuda
 * unfortunately i didn't recognize how to handle
 * matrix in form of 2 dim array
 * so i use vector
 */
int* prepare_matrix(int cols, int rows){
	int* values = new int[rows*cols];
	for(int i = 0; i<cols; i++){
		for(int j = 0; j<rows; j++){
			values[i*cols + j] = rand()%10;
		}
	}
	return values;
}


void print_vector(int* v, int len){
	for(int i = 0; i<len; i++){
		cout << v[i] << "|";
	}
	cout << endl;
}
void print_vector(double* v, int len){
	for(int i = 0; i<len; i++){
		cout << v[i] << "|";
	}
	cout << endl;
}

void print_matrix(int** m, int w, int h){
	for(int i = 0; i<h; i++){
		print_vector(m[i], w);
	}
}

void print_matrix(double** m, int w, int h){
	for(int i = 0; i<h; i++){
		print_vector(m[i], w);
	}
}

void print_matrix(matrix* matr){
	for(int i = 0; i<matr->cols; i++){
		for(int j = 0; j<matr->rows; j++){
			cout << matr->values[i*matr->cols + j] << ":";
		}
		cout << endl;
	}
}
