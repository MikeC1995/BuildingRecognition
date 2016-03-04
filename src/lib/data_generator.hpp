#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "surf.hpp"

#include <boost/python.hpp>

using namespace cv;
using namespace boost::python;


class DataGenerator
{
public:
  DataGenerator();

  void generate(const char* img_filename, const char* filenames_filename, const char* features_folder, const char* out_filename);

};
