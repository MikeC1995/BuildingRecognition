/*
** Program which accepts two images and outputs images visualising
** their matches after different stages of filtering.
*/

#include <stdio.h>
#include "/root/server/src/lib/engine.hpp"

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
  createDetector(detector, "SIFT");

  // Get object keypoints and descriptors
  std::vector<KeyPoint> objectKeypoints;
  Mat objectDescriptors;
  getKeypointsAndDescriptors(objectImage, objectKeypoints, objectDescriptors, detector);
  rootSIFT(objectDescriptors);

  // Get query keypoints and descriptors
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, detector);
  rootSIFT(queryDescriptors);

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

  // if a homography was successfully computed...
  if(homography.cols != 0 && homography.rows != 0)
  {
    std::vector<Point2f> objCorners(4);
    objCorners[0] = Point(0,0);
    objCorners[1] = Point( queryImage.cols, 0 );
    objCorners[2] = Point( queryImage.cols, queryImage.rows );
    objCorners[3] = Point( 0, queryImage.rows );
    double area = calcProjectedAreaRatio(objCorners, homography);
    std::cout << area << std::endl;
    // do not count these matches if projected area too small, (likely
    // mapping to single point => erroneous matching)
    if(area < 0.0005)
    {
      matches.clear();
    }
  }
  drawProjection(queryImage, homography, objectImage);

  Mat ransacMatchesImage;
  drawMatches(queryImage, queryKeypoints, objectImage, objectKeypoints, matches, ransacMatchesImage);
  imwrite("ransac-matches.jpg", ransacMatchesImage);
}
