#include "saveable_matcher.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <fstream>

SaveableFlannBasedMatcher::SaveableFlannBasedMatcher(const char* _filename)
{
  filename = _filename;
}

void SaveableFlannBasedMatcher::printParams()
{
    printf("SaveableFlannBasedMatcher::printParams: \n\t"
        "addedDescCount=%d\n\t"
        "flan distance_t=%d\n\t"
        "flan algorithm_t=%d\n",
        addedDescCount,
        flannIndex->getDistance(),
        flannIndex->getAlgorithm());

    std::vector<cv::String> names;
    std::vector<int> types;
    std::vector<cv::String> strValues;
    std::vector<double> numValues;

    indexParams->getAll(names, types, strValues, numValues);

    for (size_t i = 0; i < names.size(); i++)
        printf("\tindex param: %s:\t type=%d val=%s %.2f\n",
                names[i].c_str(), types[i],
                strValues[i].c_str(), numValues[i]);

    names.clear();
    types.clear();
    strValues.clear();
    numValues.clear();
    searchParams->getAll(names, types, strValues, numValues);

    for (size_t i = 0; i < names.size(); i++)
        printf("\tsearch param: %s:\t type=%d val=%s %.2f\n",
                names[i].c_str(), types[i],
                strValues[i].c_str(), numValues[i]);
}

void SaveableFlannBasedMatcher::store()
{
  // Save the matches IndexParams & SearchParams
  std::string treeFilename(filename);
  treeFilename += "-tree.xml.gz";
  cv::FileStorage store(treeFilename.c_str(), cv::FileStorage::WRITE);
  write(store);
  store.release();

  // Save the matcher index
  std::string indexFilename(filename);
  indexFilename += ".flannindex";
  writeIndex(indexFilename.c_str());

  // Save the descriptors
  std::vector<Mat> descs = getTrainDescriptors();
  std::string descriptorsFilename(filename);
  descriptorsFilename += "-descriptors.bin";
  writeDescriptors(descs, descriptorsFilename.c_str());
}

void SaveableFlannBasedMatcher::load()
{
  FILE* file; // file pointer used to check files exist

  // Read the descriptors
  std::vector<Mat> descsVec;
  std::string descriptorsFilename(filename);
  descriptorsFilename += "-descriptors.bin";
  // Check file exists
  file = fopen(descriptorsFilename.c_str(), "r");
  if(file == NULL) { return; }
  fclose(file);
  readDescriptors(descsVec, descriptorsFilename.c_str());

  // Add the descriptors to the matcher
  add(descsVec);

  std::string treeFilename(filename);
  treeFilename += "-tree.xml.gz";
  // Check file exists
  file = fopen(treeFilename.c_str(), "r");
  if(file == NULL) { return; }
  fclose(file);
  cv::FileStorage store(treeFilename.c_str(), cv::FileStorage::READ);
  cv::FileNode node = store.root();
  read(node);

  std::string indexFilename(filename);
  indexFilename += ".flannindex";
  // Check file exists
  file = fopen(indexFilename.c_str(), "r");
  if(file == NULL) { return; }
  fclose(file);
  readIndex(indexFilename.c_str());
  store.release();
}

void SaveableFlannBasedMatcher::readIndex(const char* name)
{
  indexParams->setAlgorithm(cvflann::FLANN_INDEX_SAVED);
  indexParams->setString("filename", name);

  // construct flannIndex now, so printParams works
  train();

  //printParams();
}

void SaveableFlannBasedMatcher::writeIndex(const char* name)
{
  //printParams();
  flannIndex->save(name);
}

void SaveableFlannBasedMatcher::writeDescriptors(std::vector<Mat> descriptors, const char* name)
{
  // Open the file
  std::ofstream outFILE(name, std::ios::out | std::ofstream::binary);

  // Write the number of descriptors so we can read back later
  int size = descriptors.size();
  outFILE.write(reinterpret_cast<char*>(&size), sizeof(int));

  // Write each of the descriptor matrices
  std::vector<float> descsArray;
  for(int i = 0; i < size; i++)
  {
    int width = descriptors.at(i).size().width;
    int height = descriptors.at(i).size().height;

    // Write matrix dimensions
    outFILE.write(reinterpret_cast<char*>(&width), sizeof(int));
    outFILE.write(reinterpret_cast<char*>(&height), sizeof(int));

    // Copy matrix into vector
    descsArray.assign((float*)descriptors.at(i).datastart, (float*)descriptors.at(i).dataend);

    // Finally, write actual matrix data
    outFILE.write(reinterpret_cast<char*>(&descsArray[0]), descsArray.size() * sizeof(float));
  }
  outFILE.close();
}

void SaveableFlannBasedMatcher::readDescriptors(std::vector<Mat> &descriptors, const char* name)
{
  // Open the file
  std::ifstream inFILE(name, std::ios::in | std::ios::binary);

  // Read the number of descriptors in the file
  int size;
  inFILE.read(reinterpret_cast<char*>(&size), sizeof(int));

  // Read each of the descriptor matrices
  std::vector<float> descsArray;
  for(int i = 0; i < size; i++)
  {
    int width, height;
    // Read the matrix dimensions
    inFILE.read(reinterpret_cast<char*>(&width), sizeof(int));
    inFILE.read(reinterpret_cast<char*>(&height), sizeof(int));

    // Size the descsArray to hold the data we will read
    descsArray.resize(width*height);

    // Read the data into the descsArray
    inFILE.read(reinterpret_cast<char*>(&descsArray[0]), width * height * sizeof(float));

    // Create a matrix of the correct dimensions
    Mat descsMat;
    descsMat.create(height, width, CV_32FC1);
    memcpy(descsMat.data, descsArray.data(), descsArray.size() * sizeof(float));

    // Push matrix to output descriptors vector
    descriptors.push_back(descsMat);
  }
  inFILE.close();
}
