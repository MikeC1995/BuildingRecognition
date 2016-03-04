/* Shared module to pair-wise match a query image against a set of other
** precomputed image descriptors, producing a csv of the data for analysis */

// header inclusion
#include <stdio.h>
#include <cstring>
#include "data_generator.hpp"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace cv;
using namespace boost::python;

DataGenerator::DataGenerator(){}

void DataGenerator::generate(const char* img_filename, const char* features_folder, const char* out_filename)
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

  // Open a csv file to write results to
  FILE * fp;
  fp = fopen(out_filename, "w+");

  fprintf(fp, "LAT,LNG,HEADING,#MATCHES\n");

  if(!exists(features_folder)) return;

  // Iterate over directory contents
  directory_iterator end_itr; // default directory_iterator is "end-of-directory", so use as terminator
  for(directory_iterator itr(features_folder); itr != end_itr; ++itr )
  {
    // If this is a file, not a directory...
    if (!is_directory(itr->status()))
    {
      path filename = itr->path().leaf();
      if(filename.extension().string().compare(".gz") == 0 && filename.stem().extension().string().compare(".xml") == 0)
      {
        // read the stem of the filename (without the .xml.gz extension)
        std::istringstream ss(filename.stem().stem().string());
        // vector to store the lat,lng,heading
        std::vector<std::string> vals;

        std::string token;
        while(std::getline(ss, token, ','))
        {
          vals.push_back(token);
        }

        // Read the Street View image pre-computed keypoints and descriptors
        std::string full_path(features_folder);
        full_path += "/";
        full_path += filename.string();
        FileStorage file(full_path, FileStorage::READ);
        std::vector<KeyPoint> svKeypoints;
        Mat svDescriptors;
        std::vector<Point2f> objCorners;
        file["keypoints"] >> svKeypoints;
        file["descriptors"] >> svDescriptors;
        file["objCorners"] >> objCorners;

        Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher();

        // filename for output visualisation
        //std::stringstream ss;
        //ss << "ransac-matches-" << std::to_string(i) << ".jpg";
        //std::string ransacFilename = ss.str();

        // Knn matching + Lowe filter
        std::vector<std::vector<DMatch> > knnmatches;
        std::vector<DMatch> matches;
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

        std::cout << "Image has " << matches.size() << " matches" << std::endl;
        fprintf(fp, "%s,%s,%s,%lu\n", vals.at(0).c_str(), vals.at(1).c_str(), vals.at(2).c_str(), matches.size());
      }
    }
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
