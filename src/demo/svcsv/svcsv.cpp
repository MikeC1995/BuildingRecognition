/* A program to perform RootSIFT RANSAC matching between a query image and each
** image in a specified folder. The number of matches and the visualisations
** are output. */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <string>
#include <iostream>
#include <fstream>

#include "/root/server/src/lib/surf.hpp"

using namespace cv;

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}

int main( int argc, char** argv )
{
  if(argc != 2)
  {
    DIE("Missing arguments! Usage:\n\t./svcsv <sv-folder-path>");
  }
  std::string svFolderPath(argv[1]);

  // open the file containing the list of SV filenames
  std::ifstream filenameFile;
  filenameFile.open(svFolderPath + "/filenames.txt");

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Assuming query image is saved to sv/query.jpg
  printf("Reading query image...\n");
  Mat queryImage = imread(svFolderPath + "/query.jpg");
  if(queryImage.data == NULL)
  {
    DIE("Missing query image!");
  }

  // Read the the SV image paths from file
  std::string imageFilePath;
  std::vector<std::string> svPaths;
  while(std::getline(filenameFile, imageFilePath))
  {
    svPaths.push_back(imageFilePath);
  }

  // Get query keypoints and descriptors
  printf("Computing keypoints and descriptors...\n");
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, detector);

  // Convert query keypoints to RootSIFT
  rootSIFT(queryDescriptors);

  // Open a csv file to write results to
  FILE * fp;
  fp = fopen ("out.csv", "w+");

  // Do the matching
  printf("Matching...\n");
  fprintf(fp, "LAT,LNG,HEADING,#MATCHES\n");
  Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher();
  for(int i = 0; i < svPaths.size(); i++)
  {
    // Load SV image
    Mat svImage = imread(svFolderPath + "/" + svPaths.at(i));
    if(svImage.data == NULL)
    {
      DIE("Missing image in folder!");
    }
    // SV Keypoints and descriptors
    std::vector<KeyPoint> svKeypoints;
    Mat svDescriptors;
    getKeypointsAndDescriptors(svImage, svKeypoints, svDescriptors, detector);
    // Convert SV keypoints to RootSIFT
    rootSIFT(svDescriptors);

    // filename for output visualisation
    std::stringstream ss;
    ss << "ransac-matches-" << std::to_string(i) << ".jpg";
    std::string ransacFilename = ss.str();

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
        double area = calcProjectedAreaRatio(svImage, homography);
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

  return 0;
}
