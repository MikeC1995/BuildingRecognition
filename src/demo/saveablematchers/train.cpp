/*
** Read <number> training images from <folder-name>, compute <feature-type>
** features and store the constructed matcher <matcher-name> to disk
*/

// header inclusion
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include "/root/server/src/lib/engine.hpp"
#include "/root/server/src/lib/saveable_matcher.hpp"

using namespace cv;

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}

int main( int argc, char** argv )
{
  if(argc != 5)
  {
    DIE("Missing arguments! Usage:\n\t./train <folder-name> <number> <matcher-name> <feature-type>");
  }
  std::string trainingFolderName(argv[1]);
  trainingFolderName += "/";
  int number = atoi(argv[2]);
  char* matcherName = argv[3];
  char* featureType = argv[4];
  std::string const extension = ".jpg";

  printf("Creating detector...%d\n",number);
  Ptr<FeatureDetector> detector;
  createDetector(detector, featureType);

  std::vector<std::vector<KeyPoint> > trainingKeypoints;
  std::vector<Mat> trainingDescriptors;

  for(int i = 1; i <= number; i++)
  {
    printf("Processing image %d\n", i);
    // s = image filename
    std::stringstream ss;
    ss << trainingFolderName << std::setfill('0') << std::setw(4) << std::to_string(i) << extension;
    std::string s = ss.str();

    Mat im = imread(s, 1);
    if(!im.data) {
      DIE("Missing training image data!");
    }

    // Compute keypoints and descriptors for this image
    std::vector<KeyPoint> imkps;
    detector->detect(im, imkps);
    Mat imdescs;
    detector->compute(im, imkps, imdescs);

    if(strcmp(featureType, "ROOTSIFT") == 0) rootSIFT(imdescs);

    trainingKeypoints.push_back(imkps);
    trainingDescriptors.push_back(imdescs);
  }

  //Create a matcher based on the model data
  printf("Creating matcher '%s'...\n", matcherName);
  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher(matcherName);
  printf("Adding training descriptors...\n");
  matcher->add(trainingDescriptors);
  printf("Training...\n");
  matcher->train();

  // Perform a match to actually contruct the matcher
  std::vector<DMatch> matches;
  matcher->match(trainingDescriptors.at(0), matches);

  //Save the matcher data
  printf("Saving...\n");
  matcher->store();

  printf("Done!\n");
}
