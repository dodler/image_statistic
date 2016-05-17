/*
 * test.cpp
 *
 *  Created on: May 16, 2016
 *      Author: lyan
 */

#include <png++/png.hpp>
#include "image_statistic.h"
#include <cstdlib>

using namespace std;
using namespace png;

static const char* path =
		"/media/lyan/5C88875B4189CFED/KylbergTextureDataset-1.0-png-originals.7z.001.1/KylbergTextureDataset-1.0-png-originals/blanket1-a.png";

extern "C++" double* calc_adjacency_matrix(int dm1, int dm2, int* c_values,
		int cols, int rows, int max_i, int max_j);
extern "C++" void calc_signs(double* adj_matr, int cols, int rows, int dm1,
		int dm2, int max_i, int max_j);

extern "C++" double* calc_symmetric_adjacency_matrix(int* pic, int cols,
		int rows, int dm1, int dm2, int max_i, int max_j);

int main(int argc, char* argv[]) {

	int dm1 = 5, dm2 = 5, max_i = 256, max_j = 256;

	if (argc == 3) {
		dm1 = atoi(argv[1]);
		dm2 = atoi(argv[2]);
	}

	image<rgb_pixel> image(path);
	cout << "image in memory" << endl;

	size_t height = image.get_height(), width = image.get_width();

	cout << height << endl;
	cout << width << endl;
	cout << "image dim:" << height * width << endl;

	int* vec = new int[height * width];

	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
//	         cout << (int)image[y][x].red << "," << (int)image[y][x].green << (int)image[y][x].blue << endl;
			vec[y * height + x] = (int) image[y][x].red;
		}
	}

	cout << "convertion success" << endl;

	double* matr = calc_symmetric_adjacency_matrix(vec, width, height, dm1, dm2,
			max_i, max_j);
//	print_vector(matr, width, height);
	cout << "matrix ready" << endl;
	calc_signs(matr, height, width, dm1, dm2, max_i, max_j);

	delete []matr;
	delete []vec;

	return 0;
}
