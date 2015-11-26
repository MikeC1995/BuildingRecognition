#include "saveable_matcher.hpp"

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
  descriptorsFilename += "-descriptors.xml.gz";
  cv::FileStorage descriptorsStore(descriptorsFilename.c_str(), cv::FileStorage::WRITE);
  descriptorsStore << "size" << (int)(descs.size());

  int i;
  for(i = 0; i < descs.size(); i++)
  {
    //the name of the xml node containing the descriptors
    std::stringstream sstm;
    sstm << "descriptors" << i;
    const char* descs_name = sstm.str().c_str();

    //write descriptors to the store
    descriptorsStore << descs_name << descs.at(i);
  }
  descriptorsStore.release();
}

void SaveableFlannBasedMatcher::load()
{
  // Load the descriptors
  std::vector<Mat> descsVec;
  std::string descriptorsFilename(filename);
  descriptorsFilename += ".xml.gz";
  cv::FileStorage descriptorsStore(descriptorsFilename.c_str(), cv::FileStorage::READ);

  int size;
  descriptorsStore["size"] >> size;
  printf("Reading %d descriptor matrices...\n", size);
  int i;
  for(i = 0; i < size; i++)
  {
    Mat descs;

    //the name of the xml node containing the descriptors
    std::stringstream sstm;
    sstm << "descriptors" << i;
    const char* descs_name = sstm.str().c_str();

    //load the descriptor matrix
    descriptorsStore[descs_name] >> descs;

    descsVec.push_back(descs);
  }
  descriptorsStore.release();

  // Add the descriptors to the matcher
  add(descsVec);

  std::string treeFilename(filename);
  treeFilename += "-tree.xml.gz";
  cv::FileStorage store(treeFilename.c_str(), cv::FileStorage::READ);
  cv::FileNode node = store.root();
  read(node);

  std::string indexFilename(filename);
  indexFilename += ".flannindex";
  readIndex(indexFilename.c_str());
  store.release();
}

void SaveableFlannBasedMatcher::readIndex(const char* name)
{
  indexParams->setAlgorithm(cvflann::FLANN_INDEX_SAVED);
  indexParams->setString("filename", name);

  // construct flannIndex now, so printParams works
  train();

  printParams();
}

void SaveableFlannBasedMatcher::writeIndex(const char* name)
{
  printParams();
  flannIndex->save(name);
}
