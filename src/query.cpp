// header inclusion
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <sys/time.h>
#include "lib/surf.hpp"
#include "lib/saveable_matcher.hpp"

using namespace cv;

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}

/* TODO: matches need filtering!
*  Filter before query:
*     - Filter the training keypoints + descriptors in training stage, e.g.
*       by removing duplicate keypoints in training set (see notes2.txt)
*       Check attempted implementation findGoodTrainingFeatures!
*  Filter after query, before final matching:
*     - Calculate RANSAC filter for matches corresponding to each specific training image,
*       and keep the keypoints/descriptors which pass this. The problem is that the opencv
*       match functions dont retain info about which image it matched to!
*     - Try cross check matching, i.e. match the large training set to the query image and
*       only keep those which match both ways
*     - Calculate average euclidean distance across all matches and threshold?
*     - Reject matches below a certain euclidean distance, and still threshold on number of matches?
*/

void readQueryImages(std::string const &folderpath, std::string const &extension, int number, std::vector<Mat> &images)
{
  for(int i = 1; i <= number; i++)
  {
    printf("Reading image %d\n", i);
    std::stringstream ss;
    ss << folderpath << std::setfill('0') << std::setw(4) << std::to_string(i) << extension;
    std::string s = ss.str();
    Mat im = imread(s, 1);
    images.push_back(im);
    if(!im.data) {
      DIE("Missing training image data!");
    }
  }
}

int main( int argc, char** argv )
{
  char* queryFolderName = argv[1];
  int number = atoi(argv[2]);
  strcat(queryFolderName, "/");
  std::string const extension = ".jpg";

  std::vector<Mat> queryImages;
  readQueryImages(queryFolderName, extension, number, queryImages);

  //TODO: port this to library call so same parameters used here and in training
  Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(400.0, 4, 2, 1, 0);

  // Load a matcher based on the model data
  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher("wills");
  printf("Loading matcher..\n");
  matcher->load();
  printf("Loaded!\n");


  std::vector<std::vector<DMatch> > knn_matches;
  std::vector<DMatch> matches;
  std::vector<KeyPoint> keypoints;
  struct timeval timstr;
  double tic,toc;
  for(int i = 0; i < queryImages.size(); i++)
  {
    knn_matches.clear();
    matches.clear();
    keypoints.clear();

    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    //detect keypoints and compute descriptors of query image using the detector
    Mat descriptors;
    detector->detectAndCompute(queryImages.at(i), noArray(), keypoints, descriptors, false);

    //KNN match the query images to the training set with N=2
    matcher->knnMatch(descriptors, knn_matches, 2);

    //Filter the matches according to a threshold
    loweFilter(knn_matches, matches);

    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    printf("Query %d:\t%lu matches.\t\t(Elapsed Time: %.6lf)\n", i, matches.size(), toc-tic);
  }
}
