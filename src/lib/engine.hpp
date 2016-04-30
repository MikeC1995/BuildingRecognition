#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

void createDetector(Ptr<FeatureDetector> &detector, std::string type);

void getKeypointsAndDescriptors(Mat &image, std::vector<KeyPoint> &keypoints, Mat &descriptors, Ptr<FeatureDetector> &detector);
void getKeypointsAndDescriptors(std::vector<Mat> &images, std::vector<std::vector<KeyPoint> > &keypoints, std::vector<Mat> &descriptors, Ptr<FeatureDetector> &detector);
void getKeypointsAndDescriptors(Mat &queryImage, std::vector<KeyPoint> &queryKeypoints, Mat &queryDescriptors,
  std::vector<Mat> &trainingImages, std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &trainingDescriptors,
  Ptr<FeatureDetector> &detector);

void rootSIFT(cv::Mat& descriptors);

void simpleFilter(Mat &queryDescriptors, std::vector<DMatch> &matches);
void loweFilter(std::vector<std::vector<DMatch> > &knnMatches, std::vector<DMatch> &matches);

void ransacFilter(std::vector<DMatch> &matches, std::vector<KeyPoint> &queryKeypoints, std::vector<KeyPoint> &trainingKeypoints, Mat &homography);
void ransacFilter(std::vector<DMatch> &matches, std::vector<KeyPoint> &queryKeypoints, std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &homographies);

void drawProjection(Mat &input, Mat &homography, Mat &output);
double calcProjectedAreaRatio(std::vector<Point2f> &objCorners, Mat &homography);

void getFilteredMatches(Mat &image1, std::vector<KeyPoint> &keypoints1, Mat &descriptors1, std::vector<KeyPoint> &keypoints2, Mat &descriptors2, std::vector<DMatch> &matches);
