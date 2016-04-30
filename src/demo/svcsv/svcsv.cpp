/*
** Program which produces a CSV file detailing the number of matches a query
** image makes with each image in a specified folder.
**
** Folder images given in <folder>/filenames.txt
** Query image given in <folder>/query.jpg
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <string>
#include <iostream>
#include <fstream>

#include "/root/server/src/lib/engine.hpp"

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

  // Open the file containing the list of SV filenames
  std::ifstream filenameFile;
  filenameFile.open(svFolderPath + "/filenames.txt");

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Query image is saved to sv/query.jpg
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
    std::cout << svFolderPath + "/" + svPaths.at(i) << std::endl;
    if(svImage.data == NULL)
    {
      DIE("Missing image in folder!");
    }
    // SV Keypoints and descriptors
    std::vector<KeyPoint> svKeypoints;
    Mat svDescriptors;
    getKeypointsAndDescriptors(svImage, svKeypoints, svDescriptors, detector);
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
        std::vector<Point2f> objCorners(4);
        objCorners[0] = Point(0,0);
        objCorners[1] = Point( svImage.cols, 0 );
        objCorners[2] = Point( svImage.cols, svImage.rows );
        objCorners[3] = Point( 0, svImage.rows );
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

    Mat outImg;
    drawMatches(svImage, svKeypoints, queryImage, queryKeypoints, matches, outImg);
    imwrite(ransacFilename, outImg);

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
