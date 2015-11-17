// header inclusion
#include <stdio.h>
#include <string>
#include "surf.hpp"

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
    std::stringstream ss;
    ss << folderpath << std::to_string(i) << extension;
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

  Ptr<xfeatures2d::SURF> detector;
  createSurfDetector(detector);

  std::vector<std::vector<KeyPoint> > trainingKeypoints;
  std::vector<Mat> trainingDescriptors;
  detector->detect(trainingImages, trainingKeypoints);
  detector->compute(trainingImages, trainingKeypoints, trainingDescriptors);

  // Save detector
  cv::FileStorage store("model.xml", cv::FileStorage::WRITE);
  int i;
  for(i = 0; i < trainingKeypoints.size(); i++)
  {
    //the name of the xml node containing the keypoints
    std::stringstream sstm1;
    sstm1 << "keypoints" << i;
    const char* kp_name = sstm1.str().c_str();
    //write keypoints to the store
    store << kp_name << trainingKeypoints.at(i);

    //the name of the xml node containing the descriptors
    std::stringstream sstm2;
    sstm2 << "descriptors" << i;
    const char* descs_name = sstm2.str().c_str();
    //write descriptors to the store
    store << descs_name << trainingDescriptors.at(i);
  }
  store.release();
}
