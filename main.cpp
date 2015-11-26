/////////////////////////////////////////////////////////////////////////////
//
// Individual Project
// Detect and display features in image using SURF
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <string>
#include "surf.hpp"

using namespace cv;

/* TODO:
**    -Use multiple images in matches, by combining training image keypoint vectors.
**    -Match in reverse and check for symmetry to improve good matches
**    -Compare SIFT, ASIFT, different filtering approaches etc
**    -Do good training images need to not include surrounding environment? i.e. just the building, for better matches
**        Answer = yes; then when you remove similar features amongst training images, only look at matched region
**    -Collect hundreds of high res training images of different building types (choose e.g. 5 classes of buildings, like churches, etc).
**        High res leads to more feature points, then can filter more without running out
**
**    Training set strategy:
**      -Match every keypoint in query to every keypoint in a training image.
**      -Filter with lowe, ransac etc.
**      -Then we are left with a different subset of matched keypoints in the query image for each of the training images.
**      -Could then find the intersection of these subsets to find the "good" matches,
**       or perhaps the number of matches a query keypoint is a part of is a vote of confidence in this match being "good",
**       which you could then threshold.
*/

void readTrainingImages(std::string const &folderpath, std::string const &extension, int number, std::vector<Mat> &images);
void maskMatches(std::vector<DMatch> &matches, int trainingImageIndex, std::vector<char> &mask, int &matchCount);
void ensureAtLeastKFeatures(std::vector<std::vector<KeyPoint> > &keypointList, std::vector<Mat> &descriptorsList, int k);


int main( int argc, char** argv )
{
  char* queryImageName = argv[1];
  char* trainingFolderName = argv[2];
  int number = atoi(argv[3]);
  strcat(trainingFolderName, "/");
  std::string const extension = ".jpg";

  Mat queryImage;
  std::vector<Mat> trainingImages;
  queryImage = imread( queryImageName, 1 );
  readTrainingImages(trainingFolderName, extension, number, trainingImages);

  if( argc != 4 || !queryImage.data )
  {
    printf( "Missing image data!\n" );
    return -1;
  }

  Ptr<xfeatures2d::SURF> detector;

  //Features
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  std::vector<std::vector<KeyPoint> > trainingKeypoints;
  std::vector<Mat> trainingDescriptors;

  //For each training image there is a k-length list of match sets, ranked in order of distance
  std::vector<std::vector<std::vector<DMatch> > > knnMatches;
  //For each training image there is a set of matches
  std::vector<std::vector<DMatch> > matchesSet;
  //For a single image there is a set of matches
  std::vector<DMatch> matches;

  //List of homography matrices used by RANSAC for each training image
  std::vector<Mat> homographyMatrices;

  createSurfDetector(detector);
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors,
      trainingImages, trainingKeypoints, trainingDescriptors, detector);

  //std::vector<std::vector<KeyPoint> > goodTrainingKeypoints;
  //std::vector<Mat> goodTrainingDescriptors;
  //findGoodTrainingFeatures(trainingKeypoints, trainingDescriptors, goodTrainingKeypoints, goodTrainingDescriptors);
  //trainingKeypoints = goodTrainingKeypoints;
  //trainingDescriptors = goodTrainingDescriptors;

  // Cannot match images with < k features!
  //TODO: Still get out of range errors!
  ensureAtLeastKFeatures(trainingKeypoints, trainingDescriptors, 2);

  //Fully match the query image to each of the training images
  matchKnn(queryDescriptors, trainingDescriptors, knnMatches, 2);
  for(int i = 0; i < trainingDescriptors.size(); i++)
  {
    loweFilter(knnMatches.at(i), matches);
    Mat homography;
    ransacFilter(matches, queryKeypoints, trainingKeypoints.at(i), homography);
    Mat out;
    drawMatches(queryImage, queryKeypoints, trainingImages.at(i), trainingKeypoints.at(i), matches, out,
      Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
    printf("image %d has %lu matches\n", i, matches.size());
    namedWindow("All features matching...", WINDOW_NORMAL);
    imshow("All features matching...", out);
    waitKey(0);
    matchesSet.push_back(matches);
  }

  // Filter the features in the query image according to the number of matches each keypoint makes with a set of training images
  findGoodFeatures(queryKeypoints, queryDescriptors, matchesSet, 2);

  matches.clear();
  knnMatches.clear();

  //Fully match the query image to each of the training images
  matchKnn(queryDescriptors, trainingDescriptors, knnMatches, 2);
  for(int i = 0; i < trainingDescriptors.size(); i++)
  {
    loweFilter(knnMatches.at(i), matches);
    Mat homography;
    ransacFilter(matches, queryKeypoints, trainingKeypoints.at(i), homography);
    Mat out;
    drawMatches(queryImage, queryKeypoints, trainingImages.at(i), trainingKeypoints.at(i), matches, out,
      Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
    printf("image %d has %lu matches\n", i, matches.size());
    namedWindow("Good features matching...", WINDOW_NORMAL);
    imshow("Good features matching...", out);
    waitKey(0);
  }

  return 0;
}

void ensureAtLeastKFeatures(std::vector<std::vector<KeyPoint> > &keypointList, std::vector<Mat> &descriptorsList, int k)
{
  std::vector<std::vector<KeyPoint> > newKeypointList;
  std::vector<Mat> newDescriptorsList;
  for(int i = 0; i < keypointList.size(); i++)
  {
    if(keypointList.at(i).size() >= k)
    {
      newKeypointList.push_back(keypointList.at(i));
      newDescriptorsList.push_back(descriptorsList.at(i));
    }
  }
  keypointList = newKeypointList;
  descriptorsList = newDescriptorsList;
}

void maskMatches(std::vector<DMatch> &matches, int trainingImageIndex, std::vector<char> &mask, int &matchCount)
{
  matchCount = 0;
  mask.resize(matches.size());
  fill(mask.begin(), mask.end(), 0);
  for(int i = 0; i < matches.size(); i++)
  {
    if(matches.at(i).imgIdx == trainingImageIndex)
    {
      mask.at(i) = 1;
      matchCount++;
    }
  }
}

void readTrainingImages(std::string const &folderpath, std::string const &extension, int number, std::vector<Mat> &images)
{
  for(int i = 1; i <= number; i++)
  {
    std::stringstream ss;
    ss << folderpath << std::to_string(i) << extension;
    std::string s = ss.str();
    Mat im = imread(s, 1);
    images.push_back(im);
    if(!im.data) {
      printf( "Missing training image data!\n" );
    }
  }
}
