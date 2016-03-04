/* Shared module to read an image, compute its RootSIFT keypoints and
** descriptors, and save them to a file */

// header inclusion
#include <stdio.h>
#include <cstring>
#include "feature_saver.hpp"

using namespace cv;
using namespace boost::python;

FeatureSaver::FeatureSaver(){}

void FeatureSaver::saveFeatures(const char* img_filename, const char* out_folder, const char* out_filename)
{
  Mat queryImage = imread(img_filename);
  if(queryImage.data == NULL) {
    printf("Can't read image '%s'\n", img_filename);
    return;
  }

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Get query keypoints and descriptors
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, detector);

  // Convert query keypoints to RootSIFT
  rootSIFT(queryDescriptors);

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
  file.release();
}

// Python Wrapper
BOOST_PYTHON_MODULE(feature_saver)
{
  class_<FeatureSaver>("FeatureSaver", init<>())
      .def("saveFeatures", &FeatureSaver::saveFeatures)
  ;
}
