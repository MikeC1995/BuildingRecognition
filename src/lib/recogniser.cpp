// header inclusion
#include <stdio.h>
#include "recogniser.hpp"

using namespace cv;

void query(Ptr<xfeatures2d::SURF> &detector, Ptr<SaveableFlannBasedMatcher> &matcher, Mat queryImage, std::vector<DMatch> &matches, long &original_num_matches)
{
  std::vector<std::vector<DMatch> > knn_matches;
  matches.clear();

  //detect keypoints and compute descriptors of query image using the detector
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  detector->detectAndCompute(queryImage, noArray(), keypoints, descriptors, false);

  //KNN match the query images to the training set with N=2
  matcher->knnMatch(descriptors, knn_matches, 2);

  original_num_matches = knn_matches.size();

  //Filter the matches according to a threshold
  loweFilter(knn_matches, matches);
}
