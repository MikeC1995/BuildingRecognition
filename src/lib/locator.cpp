/* Shared module which reads csv data, works out which images contained the
** subject (most matches) and from best 2 computes the lat-lng of the subject */

// header inclusion
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "locator.hpp"

using namespace cv;
using namespace boost::python;

#define PROFILE_LOCATE 1

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

// Data struc to store the vote & other data associated with a particular SV image
struct Viewpoint {
  int votes;
  std::string lat;
  std::string lng;
  std::string heading;
  std::string pitch;
  Mat image;
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
};
bool vote_sorter(Viewpoint const &lhs, Viewpoint const &rhs) {
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
  std::cout << omp_get_max_threads() << std::endl;
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
#ifdef PROFILE_LOCATE
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
  std::vector<std::vector<DMatch> > knn_matches;
  bigMatcher->knnMatch(queryDescriptors, knn_matches, 2);
  std::vector<DMatch> matches;
  loweFilter(knn_matches, matches);
#ifdef PROFILE_LOCATE
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << duration << ",";
#endif
  // Read the filenames_file to build a viewpoint table with each entry set to 0
  std::ifstream filenames_file;
  filenames_file.open(filenames_filename);
  if(!filenames_file.is_open())
  {
    return false;
  }
  std::string line;
  std::vector<Viewpoint> vpTable;
  while(std::getline(filenames_file, line))
  {
    std::vector<std::string> line_parts = splitString(line.c_str(), ',');
    Viewpoint vp;
    vp.votes = 0;
    vp.lat = line_parts.at(0);
    vp.lng = line_parts.at(1);
    vp.heading = line_parts.at(2);
    vp.pitch = line_parts.at(3);
    vpTable.push_back(vp);
  }

  // Populate the vpTable; vote for each image which a match corresponds to
  for(int i = 0; i < matches.size(); i++)
  {
    int index = matches.at(i).imgIdx;
    for(int j = 0; j < vpTable.size(); j++)
    {
      if(index == j) {
        vpTable.at(j).votes++;
      }
    }
  }

  // Sort the vpTable with the highest-matched images at the top
  std::sort(vpTable.begin(), vpTable.end(), &vote_sorter);

  // Take the top 20 of these highest-matched images
  if(vpTable.size() > 20) vpTable.resize(20);

#ifdef PROFILE_LOCATE
  // Time prep vp table
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << duration << ",";
#endif

  // Read each of these top SV images afresh to perform a rigourous matching
  std::string imgs_folder(_imgs_folder);
  #pragma omp parallel for
  for(int i = 0; i < vpTable.size(); i++)
  {
    // Read image
    Mat svImage = imread(imgs_folder + vpTable.at(i).lat + "," + vpTable.at(i).lng + "," + vpTable.at(i).heading + "," + vpTable.at(i).pitch + ".jpg");
    if(svImage.data == NULL)
    {
      printf("Unable to load SV image!\n");
      // TODO: cant break out of loop if in parallel...
      //return false;
    }
    // Get query keypoints and descriptors
    std::vector<KeyPoint> svKeypoints;
    Mat svDescriptors;
    getKeypointsAndDescriptors(svImage, svKeypoints, svDescriptors, detector);
    rootSIFT(svDescriptors);
    vpTable.at(i).image = svImage;
    vpTable.at(i).keypoints = svKeypoints;
    vpTable.at(i).descriptors = svDescriptors;

    // Match the SV image against the query, applying lowe + geometric filters
    matches.clear();
    getFilteredMatches(svImage, svKeypoints, svDescriptors, queryKeypoints, queryDescriptors, matches);

    // update the votes for this image to be the number of "rigourous" matches
    vpTable.at(i).votes = matches.size();

    // Write match images to disk
    //Mat img_matches;
    //std::stringstream ss;
    //ss << "matches" << i << ".jpg";
    //drawMatches(svImage, svKeypoints, queryImage, queryKeypoints, matches, img_matches);
    //imwrite(ss.str(), img_matches);
  }

  // Sort the vpTable again according to these new votes
  std::sort(vpTable.begin(), vpTable.end(), &vote_sorter);

#ifdef PROFILE_LOCATE
  // Time prep rigourous match
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << duration << ",";
#endif

  // Keep only the best viewpoint from each lat-lng to ensure distinct views
  std::vector<int> distinctViewIdxs;
  for(int i = 0; i < vpTable.size(); i++)
  {
    int maxIdx = i;
    for(int j = 0; j < vpTable.size(); j++)
    {
      if(i != j)
      {
        if( vpTable.at(j).lat.compare(vpTable.at(maxIdx).lat) == 0 &&
            vpTable.at(j).lng.compare(vpTable.at(maxIdx).lng) == 0 &&
            vpTable.at(j).votes > vpTable.at(maxIdx).votes )
        {
          maxIdx = j;
        }
      }
    }
    if(std::find(distinctViewIdxs.begin(), distinctViewIdxs.end(), maxIdx) == distinctViewIdxs.end()) {
      distinctViewIdxs.push_back(maxIdx);
    }
  }
  // Keep the distinct views which have at least 9 matches with the
  // query image (otherwise likely to be superfluous)
  std::vector<Viewpoint> distinctVpTable;
  for(int j = 0; j < distinctViewIdxs.size(); j++)
  {
    if(vpTable.at(distinctViewIdxs.at(j)).votes >= 9)
    {
      distinctVpTable.push_back(vpTable.at(distinctViewIdxs.at(j)));
    }
  }

  // If there are no distinct views with sufficient matches, we fail to locate the query
  if(distinctVpTable.size() == 0)
  {
    return false;
  }

  // If there's only one distinct viewpoint, use the viewpoint location as the prediction
  if(distinctVpTable.size() == 1)
  {
    lat = stod(distinctVpTable.at(0).lat);
    lng = stod(distinctVpTable.at(0).lng);
    return true;
  }

  // Match each SV image against the others
  std::vector<Viewpoint> v1s;
  std::vector<Viewpoint> v2s;
  std::vector<long> num_matches;
  for(int i = 0; i < distinctVpTable.size(); i++)
  {
    #pragma omp parallel for
    for(int j = i + 1; j < distinctVpTable.size(); j++)
    {
      matches.clear();
      Viewpoint v1 = distinctVpTable.at(i);
      Viewpoint v2 = distinctVpTable.at(j);
      getFilteredMatches(v1.image, v1.keypoints, v1.descriptors, v2.keypoints, v2.descriptors, matches);
      v1s.push_back(v1);
      v2s.push_back(v2);
      num_matches.push_back(matches.size());
    }
  }

  // Compute the intersections of each pair
  std::vector<double> lats;
  std::vector<double> lngs;
  std::vector<double> weights;
  double mean_lat = 0;
  double mean_lng = 0;
  for(int i = 0; i < v1s.size(); i++)
  {
    double x1 = stod(v1s.at(i).lng);
    double x2 = stod(v2s.at(i).lng);
    double y1 = stod(v1s.at(i).lat);
    double y2 = stod(v2s.at(i).lat);

    double alpha1 = stod(v1s.at(i).heading);
    double alpha2 = stod(v2s.at(i).heading);
    double beta1 = nfmod(90.0 - alpha1, 360.0);
    double beta2 = nfmod(90.0 - alpha2, 360.0);

    double m1 = tan(radians(90.0 - alpha1));
    double m2 = tan(radians(90.0 - alpha2));

    double x3 = (y1 - y2 + m2*x2 - m1*x1)/(m2 - m1);
    double y3 = y1 + m1*(((y1 - y2 + m2*x2 - m1*x1)/(m2 - m1))-x1);
    x3 = floor(x3 * 10000000000.0) / 10000000000.0;
    y3 = floor(y3 * 10000000000.0) / 10000000000.0;

    if(!std::isinf(x3) && !std::isinf(y3) && !std::isnan(x3) && !std::isnan(y3))
    {
      mean_lng += x3;
      mean_lat += y3;
      lngs.push_back(x3);
      lats.push_back(y3);
      weights.push_back((double)(num_matches.at(i)));
    }
  }

  // If there are no instersections, they were all parallel
  // Therefore use the best viewpoint location as the prediction
  if(lats.size() == 0)
  {
    lat = stod(distinctVpTable.at(0).lat);
    lng = stod(distinctVpTable.at(0).lng);
    return true;
  }

  // Compute means
  mean_lat /= ((double)(lats.size()));
  mean_lng /= ((double)(lngs.size()));

  // If there's only one intersection, use this as the prediction
  if(lats.size() == 1)
  {
    lat = mean_lat;
    lng = mean_lng;
    return true;
  }

  // Compute the standard deviations of the intersection coords
  double stddev_lat = 0;
  double stddev_lng = 0;
  for(int i = 0; i < weights.size(); i++)
  {
    stddev_lat += ((lats.at(i) - mean_lat)*(lats.at(i) - mean_lat));
    stddev_lat += ((lngs.at(i) - mean_lng)*(lngs.at(i) - mean_lng));
  }
  stddev_lat = sqrt(stddev_lat/(double)(lats.size() - 1));
  stddev_lng = sqrt(stddev_lng/(double)(lngs.size() - 1));

  // Remove any outlier intersections which are outside +/- 2 std. devs from mean
  for(int i = 0; i < weights.size(); i++)
  {
    if(!(lats.at(i) > mean_lat - 2 * stddev_lat && lats.at(i) < mean_lat + 2 * stddev_lat &&
       lngs.at(i) > mean_lng - 2 * stddev_lng && lngs.at(i) < mean_lng + 2 * stddev_lng))
    {
      weights.erase(weights.begin() + i);
      lats.erase(lats.begin() + i);
      lngs.erase(lngs.begin() + i);
    }
  }

  // Take the weighted average of the intersections as the prediction.
  // (Weight each intersection according to the number of matches
  // between its two viewpoints)
  double sum = 0;
  lat = 0;
  lng = 0;
  for(int i = 0; i < weights.size(); i++)
  {
    sum += weights.at(i);
  }
  for(int i = 0; i < weights.size(); i++)
  {
    weights.at(i) /= sum;
    lat += (weights.at(i) * lats.at(i));
    lng += (weights.at(i) * lngs.at(i));
  }

  #ifdef PROFILE_LOCATE
    // Time prep vp table
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
  #endif
  return true;
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
