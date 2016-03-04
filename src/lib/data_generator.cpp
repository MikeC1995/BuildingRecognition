/* Shared module to pair-wise match a query image against a set of other
** precomputed image descriptors, producing a csv of the data for analysis */

// header inclusion
#include <stdio.h>
#include <cstring>
#include <fstream>
#include "data_generator.hpp"

using namespace cv;
using namespace boost::python;

DataGenerator::DataGenerator(){}

void DataGenerator::generate(const char* img_filename, const char* filenames_filename, const char* features_folder, const char* out_filename)
{
  // open the query image
  Mat queryImage = imread(img_filename);
  if(queryImage.data == NULL) {
    printf("Can't read image '%s'\n", img_filename);
    return;
  }

  // open the file containing the list of SV filenames
  std::ifstream filenameFile;
  std::string featuresFolder(features_folder);
  std::string filenamesFilename(filenames_filename);
  filenameFile.open(filenamesFilename);

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Get query keypoints and descriptors
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, detector);

  // Convert query keypoints to RootSIFT
  rootSIFT(queryDescriptors);

  // Open a csv file to write results to and write headings
  FILE * fp;
  fp = fopen(out_filename, "w+");
  fprintf(fp, "LAT,LNG,HEADING,#MATCHES\n");

  // Read the the SV image paths from file into vector
  std::string imageFilePath;
  std::vector<std::string> svPaths;
  while(std::getline(filenameFile, imageFilePath))
  {
    svPaths.push_back(imageFilePath);
  }

  Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher();
  for(int i = 0; i < svPaths.size(); i++)
  {
    std::string full_path = featuresFolder + "/" + svPaths.at(i);
    std::cout << full_path << std::endl;

    FileStorage file(full_path, FileStorage::READ);
    std::vector<KeyPoint> svKeypoints;
    Mat svDescriptors;
    std::vector<Point2f> objCorners;
    file["keypoints"] >> svKeypoints;
    file["descriptors"] >> svDescriptors;
    file["objCorners"] >> objCorners;
    std::cout << svKeypoints.size() << " " << svDescriptors.rows << "," << svDescriptors.cols << " " << objCorners.size() << std::endl;

    // filename for output visualisation
    //std::stringstream ss;
    //ss << "ransac-matches-" << std::to_string(i) << ".jpg";
    //std::string ransacFilename = ss.str();

    // Knn matching + Lowe filter
    std::vector<std::vector<DMatch> > knnmatches;
    std::vector<DMatch> matches;
    matches.clear();
    matcher->knnMatch(svDescriptors, queryDescriptors, knnmatches, 2);
    loweFilter(knnmatches, matches);

    if(matches.size() > 4) {
      // RANSAC filter
      Mat homography;
      ransacFilter(matches, svKeypoints, queryKeypoints, homography);

      // if a homography was successfully computed...
      if(homography.cols != 0 && homography.rows != 0)
      {
        double area = calcProjectedAreaRatio(objCorners, homography);
        // do not count these matches if projected area too small, (likely
        // mapping to single point => erroneous matching)
        if(area < 0.0005)
        {
          std::cout << area << " < " << "0.0005" << std::endl;
          matches.clear();
        }
      }
    }

    // Parse latitude, longitude and heading from image filename
    std::string svPath = svPaths.at(i);
    char lat[13];
    char lng[13];
    char heading[3];
    sscanf(svPath.c_str(),"%[^','],%[^','],%[^'.']", lat, lng, heading);
    std::cout << "Image " << i << " has " << matches.size() << " matches" << std::endl;
    fprintf(fp, "%s,%s,%s,%lu\n", lat, lng, heading, matches.size());
  }

  fclose(fp);
}

// Python Wrapper
BOOST_PYTHON_MODULE(data_generator)
{
  class_<DataGenerator>("DataGenerator", init<>())
      .def("generate", &DataGenerator::generate)
  ;
}
