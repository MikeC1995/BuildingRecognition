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

//#define PROFILE_LOCATE 1

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
bool Locator::locate(const char* img_filename, const char* _imgs_folder, const char* filenames_filename)
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

  // Take the top 30 of these highest-matched images
  if(vpTable.size() > 50) vpTable.resize(50);

#ifdef PROFILE_LOCATE
  // Time prep vp table
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << duration << ",";
#endif

  // Read each of these top SV images afresh to perform a rigourous matching
  std::string imgs_folder(_imgs_folder);
  bool abort = false; // flag for omp safe loop breakout if sv image cant be read
  #pragma omp parallel for
  for(int i = 0; i < vpTable.size(); i++)
  {
    #pragma omp flush (abort)
    if (!abort) {
      // Read image
      Mat svImage = imread(imgs_folder + vpTable.at(i).lat + "," + vpTable.at(i).lng + "," + vpTable.at(i).heading + "," + vpTable.at(i).pitch + ".jpg");
      if(svImage.data == NULL)
      {
        printf("Unable to load SV image!\n");
        // set omp flag and sync across threads
        abort = true;
        #pragma omp flush (abort)
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
  }
  // An SV image couldn't be read, so we cannot locate
  if(abort)
  {
    return false;
  }

  // Sort the vpTable again according to these new votes
  std::sort(vpTable.begin(), vpTable.end(), &vote_sorter);

  std::cout << vpTable.at(0).votes << std::endl;

  // If best SV image only has 15 matches with query, probably spurious,
  // so we cannot locate.
  if(vpTable.at(0).votes < 15)
  {
    return false;
  }

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
  vpTable = distinctVpTable;

  // If there are no distinct views with sufficient matches, we fail to locate the query
  if(vpTable.size() == 0)
  {
    return false;
  }

  // If there's only one distinct viewpoint, use the viewpoint location as the prediction
  if(vpTable.size() == 1)
  {
    lat = stod(vpTable.at(0).lat);
    lng = stod(vpTable.at(0).lng);
    return true;
  }

  // Match each SV image against the others
  std::vector<Viewpoint> v1s;
  std::vector<Viewpoint> v2s;
  std::vector<std::vector<DMatch> > matchesVector;
  for(int i = 0; i < vpTable.size(); i++)
  {
    #pragma omp parallel for shared(v1s, v2s, matchesVector)
    for(int j = i + 1; j < vpTable.size(); j++)
    {
      std::vector<DMatch> vmatches;
      Viewpoint v1 = vpTable.at(i);
      Viewpoint v2 = vpTable.at(j);
      getFilteredMatches(v1.image, v1.keypoints, v1.descriptors, v2.keypoints, v2.descriptors, vmatches);
      std::vector<int> removeMatchIdxs;
      for(int m = 0; m < vmatches.size(); m++)
      {
        if(v1.keypoints.at(vmatches.at(m).queryIdx).pt.y > 610 && v2.keypoints.at(vmatches.at(m).trainIdx).pt.y > 610) {
          removeMatchIdxs.push_back(m);
        }
      }
      for(int m = 0; m < removeMatchIdxs.size(); m++)
      {
        vmatches.erase(vmatches.begin() + removeMatchIdxs.at(m));
      }

      if(vmatches.size() > 10)
      {
        #pragma omp critical
        {
          v1s.push_back(v1);
          v2s.push_back(v2);
          matchesVector.push_back(vmatches);
        }
      }
    }
  }

  // Write shortlisted viewpoints to disk for review
  for(int i = 0; i < vpTable.size(); i++)
  {
    //std::cout << vpTable.at(i).lat << "," << vpTable.at(i).lng << std::endl << std::endl;
    std::stringstream ss;
    ss << "viewpoint" << i << ".jpg";
    imwrite(ss.str(), vpTable.at(i).image);
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

    //double alpha1 = stod(v1s.at(i).heading);
    //double alpha2 = stod(v2s.at(i).heading);

    // Get average x coordinate of feature points in each image
    double avg_x1 = 0.0;
    double avg_x2 = 0.0;
    for(int k = 0; k < matchesVector.at(i).size(); k++)
    {
      avg_x1 += v1s.at(i).keypoints.at(matchesVector.at(i).at(k).queryIdx).pt.x;
    }
    for(int k = 0; k < matchesVector.at(i).size(); k++)
    {
      avg_x2 += v2s.at(i).keypoints.at(matchesVector.at(i).at(k).trainIdx).pt.x;
    }
    avg_x1 /= (double)(matchesVector.at(i).size());
    avg_x2 /= (double)(matchesVector.at(i).size());

    //double alpha1 = stod(v1s.at(i).heading) + 20.0 * (avg_x1/(double)(v1s.at(i).image.cols));
    //double alpha2 = stod(v2s.at(i).heading) + 20.0 * (avg_x2/(double)(v2s.at(i).image.cols));

    double alpha1 = stod(v1s.at(i).heading) + 10.0 * ((2 * avg_x1)/(double)(v1s.at(i).image.cols) - 1);
    double alpha2 = stod(v2s.at(i).heading) + 10.0 * ((2 * avg_x2)/(double)(v2s.at(i).image.cols) - 1);
    std::cout << alpha1 << "," << alpha2 << std::endl;


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
      weights.push_back((double)(matchesVector.at(i).size()));

      // Write match images to disk
      //Mat img_matches;
      //std::stringstream ss;
      //ss << "svmatches" << i << ".jpg";
      //drawMatches(v1s.at(i).image, v1s.at(i).keypoints, v2s.at(i).image, v2s.at(i).keypoints, matchesVector.at(i), img_matches);
      //imwrite(ss.str(), img_matches);
      //std::cout << "Mean " << i << " = " << avg_x1 << "," << avg_x2 << std::endl;
      //std::cout << v1s.at(i).lat << "," << v1s.at(i).lng << std::endl << v2s.at(i).lat << "," << v2s.at(i).lng << std::endl << y3 << "," << x3 << std::endl << matchesVector.at(i).size() << std::endl;
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
    .def("locate", &Locator::locate)
    .def("getLat", &Locator::getLat)
    .def("getLng", &Locator::getLng)
  ;
}
