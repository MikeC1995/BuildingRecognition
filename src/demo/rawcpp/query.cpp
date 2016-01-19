// header inclusion
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <sys/time.h>
#include "/root/server/src/lib/recogniser.hpp"

using namespace cv;

void DIE(const char* message)
{
  printf("%s\n", message);
  exit(1);
}

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
  char* class_name = argv[3];

  strcat(queryFolderName, "/");
  std::string const extension = ".jpg";

  std::vector<Mat> queryImages;
  readQueryImages(queryFolderName, extension, number, queryImages);

  Ptr<FeatureDetector> detector;
  createDetector(detector, "SURF");

  // Load a matcher based on the model data
  Ptr<SaveableFlannBasedMatcher> matcher = new SaveableFlannBasedMatcher("wills");
  printf("Loading matcher..\n");
  matcher->load();
  printf("Loaded!\n");

  std::vector<DMatch> matches;
  struct timeval timstr;
  double tic,toc;
  printf("Image,Class,# Descriptors,# Matches,Match time (s)\n");
  for(int i = 0; i < queryImages.size(); i++)
  {
    matches.clear();

    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    long original_num_matches;
    //TODO: below is deprecated
    //query(detector, matcher, queryImages.at(i), matches, original_num_matches);

    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    printf("%d,%s,%lu,%lu,%.6lf\n", i, class_name,original_num_matches,matches.size(),toc-tic);
  }
}
