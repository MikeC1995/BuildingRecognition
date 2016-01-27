// header inclusion
#include <stdio.h>
#include <cstring>
#include "recogniser.hpp"

using namespace cv;
using namespace boost::python;

Recogniser::Recogniser(const char* _filename, char* _featureType)
{
  filename = _filename;
  featureType = _featureType;
  printf("Creating detector...\n");
  createDetector(detector, featureType);
  printf("Created\n");
  matcher = new SaveableFlannBasedMatcher(filename);
  printf("Loading matcher '%s'...\n", filename);
  matcher->load();
  printf("Loaded!\n");
}

long Recogniser::query(const char* imagepath)
{
  Mat queryImage = imread(imagepath);

  std::vector<DMatch> matches;
  std::vector<std::vector<DMatch> > knn_matches;
  matches.clear();

  //detect keypoints and compute descriptors of query image using the detector
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  detector->detectAndCompute(queryImage, noArray(), keypoints, descriptors, false);

  if(strcmp(featureType, "ROOTSIFT") == 0) rootSIFT(descriptors);

  //KNN match the query images to the training set with N=2
  matcher->knnMatch(descriptors, knn_matches, 2);

  //Filter the matches according to a threshold
  loweFilter(knn_matches, matches);

  // Free memory
  descriptors.release();
  queryImage.release();

  return matches.size();
}

// Python Wrapper
BOOST_PYTHON_MODULE(recogniser)
{
  class_<Recogniser>("Recogniser", init<const char*, char*>())
      .def("query", &Recogniser::query)
  ;
}
