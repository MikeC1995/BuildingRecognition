#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include "saveable_matcher.hpp"
#include "surf.hpp"

#include <boost/python.hpp>

using namespace cv;
using namespace boost::python;


class FeatureSaver
{
public:
  FeatureSaver();

  void saveFeatures(const char* _img_folder, const char* _img_filenames, const char* out_folder, const char* out_filename);

};
