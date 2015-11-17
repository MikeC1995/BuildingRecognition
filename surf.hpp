#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

void createSurfDetector(Ptr<xfeatures2d::SURF> &detector, double hessianThreshold = 400.0,
    int numOctaves = 4, int numOctaveLayers = 2, int extended = 1, int upright = 0);

void getKeypointsAndDescriptors(Mat &image, std::vector<KeyPoint> &keypoints, Mat &descriptors);
void getKeypointsAndDescriptors(Mat &queryImage, std::vector<KeyPoint> &queryKeypoints, Mat &queryDescriptors,
  std::vector<Mat> &trainingImages, std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &trainingDescriptors,
  Ptr<xfeatures2d::SURF> &detector);

void simpleFilter(Mat &queryDescriptors, std::vector<DMatch> &matches);
void loweFilter(std::vector<std::vector<DMatch> > &knnMatches, std::vector<DMatch> &matches);

void ransacFilter(std::vector<DMatch> &matches, std::vector<KeyPoint> &queryKeypoints, std::vector<KeyPoint> &trainingKeypoints, Mat &homography);
void ransacFilter(std::vector<DMatch> &matches, std::vector<KeyPoint> &queryKeypoints, std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &homographies);

void findGoodFeatures(std::vector<KeyPoint> &queryKeypoints, Mat &queryDescriptors, std::vector<std::vector<DMatch> > &matchesSet, int threshold);
void findGoodTrainingFeatures(std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &trainingDescriptors, std::vector<std::vector<KeyPoint> > &goodTrainingKeypoints, std::vector<Mat> &goodTrainingDescriptors);

void drawObject(Mat &input, Mat &homography, Mat &output);
