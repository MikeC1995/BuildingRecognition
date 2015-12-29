#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "saveable_matcher.hpp"
#include "surf.hpp"

using namespace cv;

void query(Ptr<xfeatures2d::SURF> &detector, Ptr<SaveableFlannBasedMatcher> &matcher, Mat queryImage, std::vector<DMatch> &matches, long &original_num_matches);
