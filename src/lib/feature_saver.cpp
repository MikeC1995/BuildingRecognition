/* Shared module to read an image, compute its RootSIFT keypoints and
** descriptors, and save them to a file */

// header inclusion
#include <stdio.h>
#include <cstring>
#include "feature_saver.hpp"

using namespace cv;
using namespace boost::python;

FeatureSaver::FeatureSaver(){}
// separate img_filenames with ':', read each and form a small tree
void FeatureSaver::saveFeatures(const char* _img_folder, const char* _img_filenames, const char* out_folder, const char* out_filename)
{
  std::stringstream img_filenames(_img_filenames);
  std::string img_filename;
  std::vector<std::string> filename_list;
  while(std::getline(img_filenames, img_filename, ':'))
  {
     filename_list.push_back(img_filename);
  }

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
  std::stringstream first_filename(filename_list.at(0));
  std::string segment;
  std::vector<std::string> seglist;
  while(std::getline(first_filename, segment, ','))
  {
    seglist.push_back(segment);
  }

  // Create saveable matcher with name of format <lat>,<lng>
  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher((seglist.at(0) + "," + seglist.at(1)).c_str());

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Get keypoints and descriptors
  std::vector<std::vector<KeyPoint> > keypoints;
  std::vector<Mat> descriptors;
  getKeypointsAndDescriptors(images, keypoints, descriptors, detector);

  // Convert query keypoints to RootSIFT
  for(int i = 0; i < descriptors.size(); i++)
  {
    rootSIFT(descriptors.at(i));
  }

  // Build matcher tree
  matcher->add(descriptors);
  matcher->train();
  std::vector<DMatch> dummy_matches;
  matcher->match(descriptors.at(0), dummy_matches); // dummy match required for OpenCV to build tree

  // Save the matcher to disk
  matcher->store();

  // TODO: may need to store object corners?

  std::ofstream bin_file;
  bin_file.open(out_filename);
  long last_bin = 0;
  for(int i = 0; i < descriptors.size(); i++)
  {
    std::stringstream fn(filename_list.at(i));
    std::string seg;
    std::vector<std::string> list;
    while(std::getline(fn, seg, ','))
    {
      list.push_back(seg);
    }
    std::ostringstream entry;
    entry << list.at(0) << "," << list.at(1) << "," << last_bin << "," << (last_bin + descriptors.at(i).rows - 1);
    bin_file << entry.str() << std::endl;
    last_bin += descriptors.at(i).rows;
  }
  bin_file.close();

  /*

  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher();

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Get query keypoints and descriptors
  std::vector<std::vector<KeyPoint> > queryKeypoints;
  std::vector<Mat> queryDescriptors;
  getKeypointsAndDescriptors(queryImages, queryKeypoints, queryDescriptors, detector);



  // Create file to save descriptors and keypoints to
  std::string filename(out_folder);
  filename += out_filename;
  filename += ".xml.gz";
  FileStorage file(filename, FileStorage::WRITE);

  std::vector<Point2f> objCorners(4);
  objCorners[0] = Point(0,0);
  objCorners[1] = Point( queryImage.cols, 0 );
  objCorners[2] = Point( queryImage.cols, queryImage.rows );
  objCorners[3] = Point( 0, queryImage.rows );

  // Write to file
  file << "keypoints" << queryKeypoints;
  file << "descriptors" << queryDescriptors;
  file << "objCorners" << objCorners;

  // Close file
  file.release();*/
}

// Python Wrapper
BOOST_PYTHON_MODULE(feature_saver)
{
  class_<FeatureSaver>("FeatureSaver", init<>())
      .def("saveFeatures", &FeatureSaver::saveFeatures)
  ;
}
