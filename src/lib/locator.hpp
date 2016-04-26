#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "surf.hpp"
#include "saveable_matcher.hpp"

#include <boost/python.hpp>

using namespace cv;
using namespace boost::python;


class Locator
{
public:
  Locator();

  bool locateWithBigTree(const char* img_filename, const char* _imgs_folder, const char* filenames_filename);
  void locateWithCsv(const char* data_filename);

  double getLat();
  double getLng();

protected:
  Ptr<SaveableFlannBasedMatcher> bigMatcher;
  double lat;
  double lng;
};
