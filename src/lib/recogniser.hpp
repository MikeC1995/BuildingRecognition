#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "saveable_matcher.hpp"
#include "surf.hpp"

using namespace cv;

class Recogniser
{
public:
  Recogniser(const char* _filename);

  long query(Mat queryImage);
  void test();

protected:
  const char* filename;
  Ptr<SaveableFlannBasedMatcher> matcher;
  Ptr<FeatureDetector> detector;
};
