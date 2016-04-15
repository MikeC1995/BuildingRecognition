/* Shared module to read an image, compute its RootSIFT keypoints and
** descriptors, and save them to a file */

// header inclusion
#include <stdio.h>
#include <cstring>
#include "feature_saver.hpp"

using namespace cv;
using namespace boost::python;

// Split a string by the delimiter, putting each segment as an entry in the vector
std::vector<std::string> splitString(const char* str, char delimiter)
{
  std::stringstream stream(str);
  std::string segment;
  std::vector<std::string> seglist;
  while(std::getline(stream, segment, delimiter))
  {
     seglist.push_back(segment);
  }
  return seglist;
}

FeatureSaver::FeatureSaver(){}

// Store the descriptors (using a SaveableFlannBasedMatcher) for each image in _img_folder given by _img_filenames.
void FeatureSaver::saveFeatures(const char* _img_folder, const char* _img_filenames, const char* _out_folder)
{
  // separate img_filenames with ':' delimiter
  std::vector<std::string> filename_list = splitString(_img_filenames, ':');
  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // For each image, compute descriptors and save to disk
  for(int i = 0; i < filename_list.size(); i++)
  {
    std::string img_folder(_img_folder);
    Mat img = imread(img_folder + filename_list.at(i));
    if(img.data == NULL) {
      printf("Can't read image '%s'\n", filename_list.at(i).c_str());
      return;
    }

    // Get keypoints and descriptors, converting to rootSIFT
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    getKeypointsAndDescriptors(img, keypoints, descriptors, detector);
    rootSIFT(descriptors);

    // Create saveable matcher with name of format <lat>,<lng>,<heading>,<pitch>
    std::ostringstream matcher_name;
    std::string out_folder(_out_folder);
    size_t lastindex = filename_list.at(i).find_last_of(".");
    std::string rawname = filename_list.at(i).substr(0, lastindex); // remove extension
    matcher_name << out_folder << rawname;
    char* matcher_name_c = new char[matcher_name.str().size() + 1];
    strcpy(matcher_name_c, matcher_name.str().c_str()); // make copy as result of c_str() is valid only for string lifetime
    Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher(matcher_name_c);

    // Build matcher tree
    matcher->add(descriptors);
    matcher->train();
    std::vector<DMatch> dummy_matches;
    matcher->match(descriptors, dummy_matches); // dummy match to itself (required for OpenCV to build tree)
    // Save the matcher to disk
    matcher->store();
  }
}

// Read descriptors from stored SaveableFlannBasedMatchers (names given by filenames_file) and
// build a big tree from this (one big SaveableFlannBasedMatcher), saving to disk as "bigmatcher"
void FeatureSaver::saveBigTree(const char* filenames_filename, const char* folder) {
  printf("saveBigTree\n");
  // Create big matcher
  Ptr<SaveableFlannBasedMatcher> bigMatcher = new SaveableFlannBasedMatcher("bigmatcher");

  // Read each small matcher name from the filenames_file
  std::ifstream filenames_file;
  filenames_file.open(filenames_filename);
  std::string line;
  if(filenames_file.is_open())
  {
    while(std::getline(filenames_file, line))
    {
      // Load the small matcher and add its descriptors to the big matcher
      std::stringstream matcher_name;
      matcher_name << folder << line;
      printf("%s\n", matcher_name.str().c_str());
      char* matcher_name_c = new char[matcher_name.str().size() + 1];
      strcpy(matcher_name_c, matcher_name.str().c_str()); // make copy as result of c_str() is valid only for string lifetime
      Ptr<SaveableFlannBasedMatcher> smallMatcher = new SaveableFlannBasedMatcher(matcher_name_c);
      smallMatcher->load();
      std::vector<Mat> descriptors = smallMatcher->getTrainDescriptors();
      bigMatcher->add(descriptors);
    }
    printf("Training!\n");
    // Build and save the big matcher to disk
    bigMatcher->train();
    std::vector<DMatch> dummy_matches;
    printf("Matching!\n");
    bigMatcher->match(bigMatcher->getTrainDescriptors().at(0), dummy_matches); // dummy match required for OpenCV to build tree
    printf("Storing!\n");
    bigMatcher->store();
  }
}

// Python Wrapper
BOOST_PYTHON_MODULE(feature_saver)
{
  class_<FeatureSaver>("FeatureSaver", init<>())
      .def("saveFeatures", &FeatureSaver::saveFeatures)
      .def("saveBigTree", &FeatureSaver::saveBigTree)
  ;
}
