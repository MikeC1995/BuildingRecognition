// header inclusion
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "surf.hpp"

using namespace cv;

#define NUM_TRAINING_IMAGES 1

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}


int main( int argc, char** argv )
{
  char* queryImageName = argv[1];
  Mat queryImage;
  queryImage = imread( queryImageName, 1 );
  if(!queryImage.data)
  {
    DIE("Missing image data!");
  }

  printf("Reading model data...\n");
  std::vector<std::vector<KeyPoint> > trainingKeypoints;
  std::vector<Mat> trainingDescriptors;
  cv::FileStorage store("model.xml", cv::FileStorage::READ);
  int i;
  for(i = 0; i < NUM_TRAINING_IMAGES; i++)
  {
    //vars to load the keypoints + descriptors into
    std::vector<KeyPoint> kp;
    Mat descs;

    //the name of the xml node containing the keypoints
    std::stringstream sstm1;
    sstm1 << "keypoints" << i;
    const char* kp_name = sstm1.str().c_str();
    //load the keypoints
    store[kp_name] >> kp;
    trainingKeypoints.push_back(kp);

    //the name of the xml node containing the descriptors
    std::stringstream sstm2;
    sstm2 << "descriptors" << i;
    const char* descs_name = sstm2.str().c_str();
    //load the descriptors
    store[descs_name] >> descs;
    trainingDescriptors.push_back(descs);
  }
  store.release();

  printf("Successfully read %lu keypoints and %lu descriptors\n", trainingKeypoints.size(), trainingDescriptors.size());

  Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400.0, 4, 2, 1, 0);

  //detect keypoints and compute descriptors of query image using the detector
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  detector->detectAndCompute(queryImage, noArray(), keypoints, descriptors, false);

  //Create a matcher based on the model data
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
  matcher->add(trainingDescriptors);

  //KNN match the query image to the training set with N=2
  std::vector<std::vector<DMatch> > knn_matches;
  std::vector<DMatch> matches;
  matcher->knnMatch(descriptors, knn_matches, 2);

  //Filter the matches according to a threshold
  loweFilter(knn_matches, matches);

  printf("Query has %lu matches with training set\n", matches.size());
  
  /* TODO: how to filter the matches better?
  *     -Calculate RANSAC filter for matches corresponding to each specific training image?
  *     -Filter the training keypoints + descriptors in training stage
  */

  //Display matches (if single train image)
  if(NUM_TRAINING_IMAGES == 1)
  {
    Mat out;
    Mat trainingImage = imread("training/1.jpg");
    drawMatches(queryImage, keypoints, trainingImage, trainingKeypoints.at(0), matches, out,
      Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
    imshow("Output...", out);
    waitKey(0);
  }
}
