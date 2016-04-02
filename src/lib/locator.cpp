/* Shared module which reads csv data, works out which images contained the
** subject (most matches) and from best 2 computes the lat-lng of the subject */

// header inclusion
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <cmath>
#include "locator.hpp"
#include "saveable_matcher.hpp"

using namespace cv;
using namespace boost::python;

double nfmod(double a, double b)
{
    return a - b * floor(a / b);
}

double radians(double d) {
  return d * (M_PI / 180.0);
}

Locator::Locator(){}

void Locator::locateWithBigTree(const char* img_filename)
{
  Ptr<SaveableFlannBasedMatcher> bigMatcher = new SaveableFlannBasedMatcher("bigmatcher");
  bigMatcher.load();


}

void Locator::locate(const char* data_filename)
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
      .def("locate", &Locator::locate)
      .def("getLat", &Locator::getLat)
      .def("getLng", &Locator::getLng)
  ;
}
