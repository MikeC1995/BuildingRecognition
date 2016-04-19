/* Shared module which reads csv data, works out which images contained the
** subject (most matches) and from best 2 computes the lat-lng of the subject */

// header inclusion
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <cmath>
#include "locator.hpp"

using namespace cv;
using namespace boost::python;

double nfmod(double a, double b)
{
    return a - b * floor(a / b);
}

double radians(double d) {
  return d * (M_PI / 180.0);
}

Locator::Locator() {
  // Load the big matcher
  bigMatcher = new SaveableFlannBasedMatcher("bigmatcher");
  bigMatcher->load();
}

// Data struc to store the vote associated with a particular SV image
struct vote {
  std::string lat;
  std::string lng;
  std::string heading;
  std::string pitch;
  int votes;
};
bool vote_sorter(vote const &lhs, vote const &rhs) {
  return lhs.votes > rhs.votes; // sorts in descending order
}

// Split a string by the delimiter, putting each segment as an entry in the vector
std::vector<std::string> splitString(const char* str, char delimiter)
{
  std::stringstream stream(str);
  std::string segment;
  std::vector<std::string> seglist;
  while(std::getline(stream, segment, delimiter))
  {
     seglist.push_back(segment);
  }
  return seglist;
}

// Locate the object in the image given by img_filename by matching against the stored bigmatcher,
// taking the top scoring images, and performing a rigourous matching against these.
// (_imgs_folder = the folder containing the SV images, filenames_filename = the location of the file describing the SV filenames)
bool Locator::locateWithBigTree(const char* img_filename, const char* _imgs_folder, const char* filenames_filename)
{
  // Load the query image
  Mat queryImage = imread(img_filename);
  if(queryImage.data == NULL)
  {
    printf("Can't read image '%s'\n", img_filename);
    return false;
  }

  // Create SIFT detector
  Ptr<FeatureDetector> detector;
  createDetector(detector, "SIFT");

  // Get query keypoints and descriptors, converting to rootSIFT
  std::vector<KeyPoint> queryKeypoints;
  Mat queryDescriptors;
  getKeypointsAndDescriptors(queryImage, queryKeypoints, queryDescriptors, detector);
  rootSIFT(queryDescriptors);

  // Match query image against all SV images using bigmatcher
  std::vector<std::vector<DMatch> > knn_matches;
  bigMatcher->knnMatch(queryDescriptors, knn_matches, 2);
  std::vector<DMatch> matches;
  loweFilter(knn_matches, matches);

  // Read the filenames_file to build a vote table with each entry set to 0
  std::ifstream filenames_file;
  filenames_file.open(filenames_filename);
  if(!filenames_file.is_open())
  {
    return false;
  }
  std::string line;
  std::vector<vote> voteTable;
  while(std::getline(filenames_file, line))
  {
    std::vector<std::string> line_parts = splitString(line.c_str(), ',');
    vote data;
    data.lat = line_parts.at(0);
    data.lng = line_parts.at(1);
    data.heading = line_parts.at(2);
    data.pitch = line_parts.at(3);
    data.votes = 0;
    voteTable.push_back(data);
  }

  // Populate the voteTable; vote for each image which a match corresponds to
  for(int i = 0; i < matches.size(); i++)
  {
    int index = matches.at(i).imgIdx;
    for(int j = 0; j < voteTable.size(); j++)
    {
      if(index == j) {
        voteTable.at(j).votes++;
      }
    }
  }

  // Sort the voteTable with the highest-matched images at the top
  std::sort(voteTable.begin(), voteTable.end(), &vote_sorter);

  // Take the top 10 of these highest-matched images
  if(voteTable.size() > 10) voteTable.resize(10);
  // Read each of these top SV images afresh to perform a rigourous matching
  std::string imgs_folder(_imgs_folder);
  for(int i = 0; i < voteTable.size(); i++)
  {
    // Read image
    Mat svImage = imread(imgs_folder + voteTable.at(i).lat + "," + voteTable.at(i).lng + "," + voteTable.at(i).heading + "," + voteTable.at(i).pitch + ".jpg");
    if(svImage.data == NULL)
    {
      printf("Unable to load sv image!\n");
      return false;
    }
    // Get query keypoints and descriptors
    std::vector<KeyPoint> svKeypoints;
    Mat svDescriptors;
    getKeypointsAndDescriptors(svImage, svKeypoints, svDescriptors, detector);
    rootSIFT(svDescriptors);

    // Match as standard
    Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher();
    matches.clear();
    knn_matches.clear();
    matcher->knnMatch(svDescriptors, queryDescriptors, knn_matches, 2);
    loweFilter(knn_matches, matches);

    // Perform geometric verification
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
          matches.clear();
        }
      }
    }
    // update the votes for this image to be the number of "rigourous" matches
    voteTable.at(i).votes = matches.size();

    Mat img_matches;
    std::stringstream ss;
    ss << "matches" << i << ".jpg";
    drawMatches(svImage, svKeypoints, queryImage, queryKeypoints, matches, img_matches);
    imwrite(ss.str(), img_matches);
  }
  // Sort the vote table again according to these new votes
  std::sort(voteTable.begin(), voteTable.end(), &vote_sorter);
  for(int i = 0; i < voteTable.size(); i++) { std::cout << voteTable.at(i).votes << std::endl; }

  for(int i = 1; i < voteTable.size(); i++)
  {
    int j = i - 1;
    // less than 10 matches is probably superfluous matches, so report no object found
    if(voteTable.at(j).votes < 10 && voteTable.at(i).votes < 10)
    {
      printf("Not enough votes!\n");
      return false;
    }

    // Take the top scoring images to perform the triangulation
    double x1 = stod(voteTable.at(j).lng);
    double x2 = stod(voteTable.at(i).lng);
    double y1 = stod(voteTable.at(j).lat);
    double y2 = stod(voteTable.at(i).lat);

    double alpha1 = stod(voteTable.at(j).heading);
    double alpha2 = stod(voteTable.at(i).heading);
    double beta1 = nfmod(90.0 - alpha1, 360.0);
    double beta2 = nfmod(90.0 - alpha2, 360.0);
    double a =
      ((x2-x1)*sin(radians(beta2)) - (y2-y1)*cos(radians(beta2))) /
      (cos(radians(beta1))*sin(radians(beta2)) - sin(radians(beta1))*cos(radians(beta2)));

    double x3 = x1 + a*cos(radians(beta1));
    double y3 = y1 + a*sin(radians(beta1));

    lng = floor(x3 * 10000000000.0) / 10000000000.0;
    lat = floor(y3 * 10000000000.0) / 10000000000.0;
    if(!std::isinf(lng) && !std::isinf(lat) && !std::isnan(lat) && !std::isnan(lng))
    {
      return true;
    }
  }
  return false;
}

void Locator::locateWithCsv(const char* data_filename)
{
  std::ifstream file(data_filename);

  std::string line;
  std::string last_lat;
  std::string last_lng;
  std::vector<double> latitudes;
  std::vector<double> longitudes;
  std::vector<int> maxMatches;
  std::vector<double> maxHeadings;
  int maxMatch = -1;
  double maxHeading = -1.0;
  int i = 0;
  while(std::getline(file, line))
  {
    if(line.compare("LAT,LNG,HEADING,#MATCHES") == 0)
    {
      continue;
    }
    std::stringstream ss(line);
    std::string value;
    std::vector<std::string> values;
    while(std::getline(ss, value, ','))
    {
      values.push_back(value);

      if(ss.peek() == ',')
      {
        ss.ignore();
      }
    }

    if(values.at(0).compare(last_lat) == 0 && values.at(1).compare(last_lng) == 0)
    {
      if(std::stoi(values.at(3)) > maxMatch)
      {
        maxHeading = std::stod(values.at(2));
        maxMatch = std::stoi(values.at(3));
      }
    } else {
      maxMatches.push_back(maxMatch);
      maxHeadings.push_back(maxHeading);
      latitudes.push_back(std::stod(values.at(0)));
      longitudes.push_back(std::stod(values.at(1)));

      last_lat = values.at(0);
      last_lng = values.at(1);
      maxHeading = std::stod(values.at(2));
      maxMatch = std::stoi(values.at(3));
    }
  }

  std::vector<int> sortedIndices(maxMatches.size());
  std::size_t n(0);
  std::generate(std::begin(sortedIndices), std::end(sortedIndices), [&]{ return n++; });

  std::sort(  std::begin(sortedIndices),
              std::end(sortedIndices),
              [&](int i1, int i2) { return maxMatches[i1] < maxMatches[i2]; } );

  double maxIdx = sortedIndices.at(sortedIndices.size() - 1);
  double secondmaxIdx;
  for(int i = sortedIndices.size() - 2; i >= 0; i--)
  {
    if(maxHeadings.at(sortedIndices.at(i)) != maxHeadings.at(maxIdx))
    {
      secondmaxIdx = sortedIndices.at(i);
      break;
    }
  }

  double x1 = longitudes.at(maxIdx);
  double x2 = longitudes.at(secondmaxIdx);
  double y1 = latitudes.at(maxIdx);
  double y2 = latitudes.at(secondmaxIdx);

  double alpha1 = maxHeadings.at(maxIdx);
  double alpha2 = maxHeadings.at(secondmaxIdx);
  double beta1 = nfmod(90.0 - alpha1, 360.0);
  double beta2 = nfmod(90.0 - alpha2, 360.0);
  double a =
    ((x2-x1)*sin(radians(beta2)) - (y2-y1)*cos(radians(beta2))) /
    (cos(radians(beta1))*sin(radians(beta2)) - sin(radians(beta1))*cos(radians(beta2)));

  double x3 = x1 + a*cos(radians(beta1));
  double y3 = y1 + a*sin(radians(beta1));

  lng = floor(x3 * 10000000000.0) / 10000000000.0;
  lat = floor(y3 * 10000000000.0) / 10000000000.0;
}

double Locator::getLat() {
  return lat;
}

double Locator::getLng() {
  return lng;
}

// Python Wrapper
BOOST_PYTHON_MODULE(locator)
{
  class_<Locator>("Locator", init<>())
    .def("locateWithBigTree", &Locator::locateWithBigTree)
    .def("locateWithCsv", &Locator::locateWithCsv)
    .def("getLat", &Locator::getLat)
    .def("getLng", &Locator::getLng)
  ;
}
