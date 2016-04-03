/* Shared module to read an image, compute its RootSIFT keypoints and
** descriptors, and save them to a file */

// header inclusion
#include <stdio.h>
#include <cstring>
#include "feature_saver.hpp"

using namespace cv;
using namespace boost::python;

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

void FeatureSaver::saveFeatures(const char* _img_folder, const char* _img_filenames, const char* _out_folder, const char* out_filename)
{
  // TODO: may need to store object corners?

  // separate img_filenames with ':' delimiter
  std::vector<std::string> filename_list = splitString(_img_filenames, ':');

  // read each image and push to images vector
  std::vector<Mat> images;
  for(int i = 0; i < filename_list.size(); i++)
  {
    std::string img_folder(_img_folder);
    Mat img = imread(img_folder + filename_list.at(i));
    if(img.data == NULL) {
      printf("Can't read image '%s'\n", filename_list.at(i).c_str());
      return;
    }
    images.push_back(img);
  }

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Get keypoints and descriptors
  std::vector<std::vector<KeyPoint> > keypoints;
  std::vector<Mat> descriptors;
  getKeypointsAndDescriptors(images, keypoints, descriptors, detector);

  // Convert query descriptors to RootSIFT and save to disk
  for(int i = 0; i < descriptors.size(); i++)
  {
    rootSIFT(descriptors.at(i));

    // Create saveable matcher with name of format <lat>,<lng>,<heading>
    std::ostringstream matcher_name;
    std::string out_folder(_out_folder);
    size_t lastindex = filename_list.at(i).find_last_of(".");
    std::string rawname = filename_list.at(i).substr(0, lastindex); // remove extension
    matcher_name << out_folder << rawname;
    char* matcher_name_c = new char[matcher_name.str().size() + 1];
    strcpy(matcher_name_c, matcher_name.str().c_str()); // make copy as result of c_str() is valid only for string lifetime
    Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher(matcher_name_c);

    // Build matcher tree
    matcher->add(descriptors.at(i));
    matcher->train();
    std::vector<DMatch> dummy_matches;
    matcher->match(descriptors.at(i), dummy_matches); // dummy match required for OpenCV to build tree

    // Save the matcher to disk
    matcher->store();
  }

  // Write to the bins file, with format <lat>,<lng>,<start-bin>,<end-bin>:
  // Read last bin written, this is the first_bin
  std::ifstream bin_file_in;
  bin_file_in.open(out_filename);
  std::string line, last_line;
  if(bin_file_in.is_open())
  {
    while(std::getline(bin_file_in, line))
    {
      last_line = line;
    }
  }
  long first_bin;
  if(splitString(last_line.c_str(), ',').size() > 0)
  {
    first_bin = stol(splitString(last_line.c_str(), ',').at(3)) + 1;
  } else {
    first_bin = 0;
  }
  bin_file_in.close();

  // Count # descriptors in each image for this lat-lng and add together to find this bin width
  // Write the last bin to the file
  std::ofstream bin_file_out;
  bin_file_out.open(out_filename, std::ios::app);
  if(bin_file_out.is_open())
  {
    long last_bin = first_bin;
    for(int i = 0; i < descriptors.size(); i++)
    {
      std::vector<std::string> list = splitString(filename_list.at(i).c_str(), ',');
      std::ostringstream entry;
      entry << list.at(0) << "," << list.at(1) << "," << list.at(2) << "," << first_bin << "," << (last_bin + 1);
      bin_file_out << entry.str() << std::endl;
      last_bin += 1;
    }
    bin_file_out.close();
  }
}

void FeatureSaver::saveBigTree(const char* filenames_filename, const char* folder) {
  Ptr<SaveableFlannBasedMatcher> bigMatcher = new SaveableFlannBasedMatcher("bigmatcher");

  std::ifstream filenames_file;
  filenames_file.open(filenames_filename);
  std::string line;
  if(filenames_file.is_open())
  {
    while(std::getline(filenames_file, line))
    {
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
    bigMatcher->train();
    std::vector<DMatch> dummy_matches;
    bigMatcher->match(bigMatcher->getTrainDescriptors().at(0), dummy_matches); // dummy match required for OpenCV to build tree
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
