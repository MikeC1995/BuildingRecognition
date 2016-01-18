#include <stdio.h>
#include "../lib/surf.hpp"

using namespace cv;

int main( int argc, char** argv )
{
  char* objectImageName = argv[1];
  char* queryImageName = argv[2];

  Mat objectImage = imread(objectImageName);
  Mat queryImage = imread(queryImageName);

  if(objectImage.data == NULL || queryImage.data == NULL)
  {
    printf("Missing image data!\n");
    exit(1);
  }

  Ptr<FeatureDetector> detector;
  createDetector(detector, "SURF");

  // Get object key points and descriptors
  std::vector<KeyPoint> objectKeypoints;
  Mat objectDescriptors;
  getKeypointsAndDescriptors(objectImage, objectKeypoints, objectDescriptors, detector);

  // Get query keypoints and descriptors
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, detector);

  // Write keypoint visualisation images
  Mat objectKeypointsImage;
  drawKeypoints(objectImage, objectKeypoints, objectKeypointsImage);
  Mat queryKeypointsImage;
  drawKeypoints(queryImage, queryKeypoints, queryKeypointsImage);
  imwrite("object-keypoints.jpg", objectKeypointsImage);
  imwrite("query-keypoints.jpg", queryKeypointsImage);

  Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher();

  // Single matching
  std::vector<DMatch> matches;
  Mat singleMatchesImage;
  matcher->match(queryDescriptors, objectDescriptors, matches);
  drawMatches(queryImage, queryKeypoints, objectImage, objectKeypoints, matches, singleMatchesImage);
  imwrite("single-matches.jpg", singleMatchesImage);

  // Knn matching + Lowe filter
  std::vector<std::vector<DMatch> > knnmatches;
  matches.clear();
  matcher->knnMatch(queryDescriptors, objectDescriptors, knnmatches, 2);
  loweFilter(knnmatches, matches);
  Mat loweMatchesImage;
  drawMatches(queryImage, queryKeypoints, objectImage, objectKeypoints, matches, loweMatchesImage);
  imwrite("lowe-matches.jpg", loweMatchesImage);

  // RANSAC filter
  Mat homography;
  ransacFilter(matches, queryKeypoints, objectKeypoints, homography);
  Mat ransacMatchesImage;
  drawMatches(queryImage, queryKeypoints, objectImage, objectKeypoints, matches, ransacMatchesImage);
  imwrite("ransac-matches.jpg", ransacMatchesImage);
}
