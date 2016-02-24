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

int main( int argc, char** argv )
{
  if(argc != 6)
  {
    DIE("Missing arguments! Usage:\n\t./query <folder-name> <number> <class-name> <matcher-name> <feature-type>");
  }
  std::string queryFolderName(argv[1]);
  queryFolderName += "/";
  int number = atoi(argv[2]);
  char* className = argv[3];
  char* matcherName = argv[4];
  char* featureType = argv[5];
  std::string const extension = ".jpg";

  Recogniser r = *(new Recogniser(matcherName, featureType));

  struct timeval timstr;
  double tic,toc;

  printf("Image | Class | #Matches | Match time (s)\n");
  for(int i = 1; i <= number; i++)
  {
    // s = image filename
    std::stringstream ss;
    ss << queryFolderName << std::setfill('0') << std::setw(4) << std::to_string(i) << extension;
    std::string s = ss.str();

    // start timer
    gettimeofday(&timstr,NULL);
    tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    long matches = r.query(s.c_str());

    // end timer
    gettimeofday(&timstr,NULL);
    toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);

    printf("%d,%s,%lu,%.6lf\n", i, className,matches,toc-tic);
  }
}
