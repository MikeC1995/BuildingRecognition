/* A program to perform RootSIFT RANSAC matching between a query image and each
** image in a specified folder. The number of matches and the visualisations
** are output. */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <string>

#include "/root/server/src/lib/surf.hpp"

using namespace cv;

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}

int main( int argc, char** argv )
{
  if(argc != 4)
  {
    DIE("Missing arguments! Usage:\n\t./simplematch <query-image> <folder> <number>");
  }
  std::string queryPath(argv[1]);
  std::string folderPath(argv[2]);
  folderPath += "/";
  int number = atoi(argv[3]);
  std::string const extension = ".jpg";

  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Read the query image
  printf("Reading query image...\n");
  Mat queryImage = imread(queryPath.c_str());
  if(queryImage.data == NULL)
  {
    DIE("Missing query image!");
  }

  // Read each image in folder
  printf("Reading folder images...\n");
  std::vector<Mat> folderImages;
  for(int i = 1; i <= number; i++)
  {
    std::stringstream ss;
    ss << folderPath << std::setfill('0') << std::setw(4) << std::to_string(i) << extension;
    std::string s = ss.str();

    // Read the folder image
    Mat img = imread(s.c_str());
    if(img.data == NULL)
    {
      DIE("Missing image in folder!");
    }
    folderImages.push_back(img);
  }

  // Get query keypoints and descriptors
  printf("Computing keypoints and descriptors...\n");
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  std::vector<std::vector<KeyPoint> > folderKeypoints;
  std::vector<Mat> folderDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, folderImages, folderKeypoints, folderDescriptors, detector);

  // Convert to RootSIFT
  printf("Converting to RootSIFT...\n");
  rootSIFT(queryDescriptors);
  for(int i = 0; i < folderDescriptors.size(); i++)
  {
    rootSIFT(folderDescriptors.at(i));
  }

  // Do the matching
  printf("Matching...\n");
  Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher();
  for(int i = 0; i < folderImages.size(); i++)
  {
    // filenames
    std::stringstream ss;
    ss << "lowe-matches-" << std::to_string(i) << ".jpg";
    std::string loweFilename = ss.str();
    ss.str("");
    ss << "ransac-matches-" << std::to_string(i) << ".jpg";
    std::string ransacFilename = ss.str();

    // Knn matching + Lowe filter
    std::vector<std::vector<DMatch> > knnmatches;
    std::vector<DMatch> matches;
    matches.clear();
    matcher->knnMatch(queryDescriptors, folderDescriptors.at(i), knnmatches, 2);
    loweFilter(knnmatches, matches);
    Mat loweMatchesImage;
    drawMatches(queryImage, queryKeypoints, folderImages.at(i), folderKeypoints.at(i), matches, loweMatchesImage);
    imwrite(loweFilename.c_str(), loweMatchesImage);

    if(matches.size() > 4) {
      // RANSAC filter
      Mat homography;
      ransacFilter(matches, queryKeypoints, folderKeypoints.at(i), homography);
      Mat ransacMatchesImage;
      drawMatches(queryImage, queryKeypoints, folderImages.at(i), folderKeypoints.at(i), matches, ransacMatchesImage);
      imwrite(ransacFilename.c_str(), ransacMatchesImage);
    }
    printf("Image %d: %lu\n", i, matches.size());
  }

  return 0;
}
