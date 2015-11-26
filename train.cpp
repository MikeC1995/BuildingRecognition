// header inclusion
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include "surf.hpp"
#include "saveable_matcher.hpp"

using namespace cv;

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}


void readTrainingImages(std::string const &folderpath, std::string const &extension, int number, std::vector<Mat> &images)
{
  for(int i = 1; i <= number; i++)
  {
    printf("Reading image %d\n", i);
    std::stringstream ss;
    ss << folderpath << std::setfill('0') << std::setw(4) << std::to_string(i) << extension;
    std::string s = ss.str();
    Mat im = imread(s, 1);
    images.push_back(im);
    if(!im.data) {
      DIE("Missing training image data!");
    }
  }
}

int main( int argc, char** argv )
{
  char* trainingFolderName = argv[1];
  int number = atoi(argv[2]);
  strcat(trainingFolderName, "/");
  std::string const extension = ".jpg";

  std::vector<Mat> trainingImages;
  readTrainingImages(trainingFolderName, extension, number, trainingImages);

  printf("Creating detector...\n");
  Ptr<xfeatures2d::SURF> detector;
  createSurfDetector(detector);

  std::vector<std::vector<KeyPoint> > trainingKeypoints;
  std::vector<Mat> trainingDescriptors;
  printf("Detecting keypoints...\n");
  detector->detect(trainingImages, trainingKeypoints);
  printf("Computing descriptors...\n");
  detector->compute(trainingImages, trainingKeypoints, trainingDescriptors);

  //Create a matcher based on the model data
  printf("Creating matcher...\n");
  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher("wills");
  printf("Adding training descriptors...\n");
  matcher->add(trainingDescriptors);
  printf("Training...\n");
  matcher->train();

  // Perform a match to actually contruct the matcher
  // See http://stackoverflow.com/questions/9248012/saving-and-loading-flannbasedmatcher?rq=1
  // ...see if you can get around this using the derived class workaround
  std::vector<DMatch> matches;
  matcher->match(trainingDescriptors.at(0), matches);

  //Save the matcher data
  printf("Saving matcher...\n");
  matcher->store();

  printf("Done!\n");
}
