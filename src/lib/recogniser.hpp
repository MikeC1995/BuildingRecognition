#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "saveable_matcher.hpp"
#include "engine.hpp"

#include <boost/python.hpp>

using namespace cv;
using namespace boost::python;


class Recogniser
{
public:
  Recogniser(const char* _filename, char* featureType);

  long query(const char* imagepath);

protected:
  const char* filename;
  char* featureType;
  Ptr<SaveableFlannBasedMatcher> matcher;
  Ptr<FeatureDetector> detector;
};
