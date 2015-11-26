// header inclusion
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "surf.hpp"
#include "saveable_matcher.hpp"

using namespace cv;

#define NUM_TRAINING_IMAGES 30

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}

/* TODO: matches need filtering!
*  Filter before query:
*     - Filter the training keypoints + descriptors in training stage, e.g.
*       by removing duplicate keypoints in training set (see notes2.txt)
*       Check attempted implementation findGoodTrainingFeatures!
*  Filter after query, before final matching:
*     - Calculate RANSAC filter for matches corresponding to each specific training image,
*       and keep the keypoints/descriptors which pass this. The problem is that the opencv
*       match functions dont retain info about which image it matched to!
*     - Try cross check matching, i.e. match the large training set to the query image and
*       only keep those which match both ways
*/

int main( int argc, char** argv )
{
  char* queryImageName = argv[1];
  Mat queryImage;
  queryImage = imread( queryImageName, 1 );
  if(!queryImage.data)
  {
    DIE("Missing image data!");
  }

  Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400.0, 4, 2, 1, 0);

  //detect keypoints and compute descriptors of query image using the detector
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  detector->detectAndCompute(queryImage, noArray(), keypoints, descriptors, false);

  //Create a matcher based on the model data
  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher("wills");
  printf("Loading matcher..\n");
  matcher->load();
  printf("Loaded!\n");

  //KNN match the query image to the training set with N=2
  std::vector<std::vector<DMatch> > knn_matches;
  std::vector<DMatch> matches;
  matcher->knnMatch(descriptors, knn_matches, 2);

  //Filter the matches according to a threshold
  loweFilter(knn_matches, matches);

  printf("Query has %lu matches with training set\n", matches.size());
}
