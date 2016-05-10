#define cimg_display 0
#include <png++/png.hpp>
#include <iostream>
#include "image_statistic.h"

using namespace std;
using namespace png;

const static char* image_source_path =
		"/media/dodler/CEE4E048E4E033FD/samples/KylbergTextureDataset-1.0-png-originals/blanket2-a.png";

int main(void){
	png::image< png::rgb_pixel > img(image_source_path);

	size_t w = img.get_width(), h = img.get_height();
	cout << w << "|" << h << endl;

//	int* buf = new int[];
}
