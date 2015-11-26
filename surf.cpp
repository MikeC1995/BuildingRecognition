/////////////////////////////////////////////////////////////////////////////
//
// Individual Project
// Detect and display features in image using SURF
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <string>
#include "surf.hpp"

using namespace cv;

/* Create a SURF keypoint detector.
**
**    Out:  detector
*/
void createSurfDetector(Ptr<xfeatures2d::SURF> &detector, double hessianThreshold,
    int numOctaves, int numOctaveLayers, int extended, int upright)
{
  /*  Parameter descriptions
  **
  **  hessianThreshold
  **  The higher the threshold, fewer keypoints are detected
  **  Too low a value = weak feature points which have less repeatability.
  **  Too high a value = not be enough features to describe the image.
  **
  **  numOctaves
  **  Determines the size of the features the detector looks for.
  **  For large features, use a larger value.
  **
  **  numOctaveLayers
  **  Determines the range of feature sizes that can be detected.
  **  More layers = detect features of many different sizes
  **
  **  extended
  **  0 = 64 dimension descriptors; 1 = 128 dimensions
  **
  **  upright
  **  0 = compute orientation of each feature; 1 = do not compute orientation
  */

  detector = xfeatures2d::SURF::create(hessianThreshold, numOctaves, numOctaveLayers, extended, upright);
}

/* Create a SURF feature detector, and using it calculate keypoints and
** descriptors the input images.
*/
/* Single image.
**  In:   image
**  Out:  keypoints, descriptors
*/
void getKeypointsAndDescriptors(Mat &image, std::vector<KeyPoint> &keypoints, Mat &descriptors)
{
  //The higher the threshold, fewer keypoints are detected
  //Too low a value = weak feature points which have less repeatability.
  //Too high a value = not be enough features to describe the image.
  double hessianThreshold = 400.0;

  //Determines the size of the features the detector looks for.
  //For large features, use a larger value.
  int numOctaves = 4;

  //Determines the range of feature sizes that can be detected.
  //More layers = detect features of many different sizes
  int numOctaveLayers = 2;

  //0 = 64 dimension descriptors; 1 = 128 dimensions
  int extended = 1;

  //0 = compute orientation of each feature; 1 = do not compute orientation
  int upright = 0;

  Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(hessianThreshold, numOctaves, numOctaveLayers, extended, upright);

  //detect keypoints and compute descriptors using the detector
  detector->detectAndCompute(image, noArray(), keypoints, descriptors, false);

  waitKey(0);
}

/* Single query image, multiple training images.
**  In:   queryImage, trainingImages, detector
**  Out:  queryKeypoints, queryDescriptors, trainingKeypoints, trainingDescriptors
*/
void getKeypointsAndDescriptors(Mat &queryImage, std::vector<KeyPoint> &queryKeypoints, Mat &queryDescriptors,
  std::vector<Mat> &trainingImages, std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &trainingDescriptors,
  Ptr<xfeatures2d::SURF> &detector)
{
  detector->detect(queryImage, queryKeypoints);
  detector->detect(trainingImages, trainingKeypoints);
  detector->compute(queryImage, queryKeypoints, queryDescriptors);
  detector->compute(trainingImages, trainingKeypoints, trainingDescriptors);
}

/* Filter a set of matches by thresholding at twice the minimum distance,
** (i.e. twice the best-match distance)
**
**    In:   queryDescriptors
**    Out:  matches
*/
void simpleFilter(Mat &queryDescriptors, std::vector<DMatch> &matches)
{
  //Calculate best match distance
  double min_dist = matches[0].distance;
  for(int i = 0; i < queryDescriptors.rows; i++)
  {
    double dist = matches[i].distance;
    if(dist < min_dist) min_dist = dist;
  }

  // Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  // or a small arbitary value ( 0.02 ) in the event that min_dist is very
  // small)
  std::vector<DMatch> good_matches;
  for(int i = 0; i < queryDescriptors.rows; i++)
  {
    if(matches[i].distance <= max(2*min_dist, 0.02))
    {
      good_matches.push_back(matches[i]);
    }
  }
  matches = good_matches;
}

/* Filter a set of knn matches according to Lowe's
** nearest neighbour distance ratio.
**
**    In:   knnMatches
**    Out:  matches
*/
void loweFilter(std::vector<std::vector<DMatch> > &knnMatches, std::vector<DMatch> &matches)
{
  std::vector<DMatch> good_matches;
  int count = 0;
  for (int i = 0; i < knnMatches.size(); i++)
  {
    const float ratio = 0.8; // 0.8 in Lowe's paper; can be tuned
    if (knnMatches[i][0].distance <= ratio * knnMatches[i][1].distance)
    {
      good_matches.push_back(knnMatches[i][0]);
    } else {
      count++;
    }
  }
  //printf("Lowe filter removed %d matches.\n", count);
  matches = good_matches;
}


/* Filter a set of matches by computing the homography matrix describing the transform
** of the query keypoints onto the training keypoints. Internally, this computation uses
** RANSAC to find inliers, and inlier matches are the ones which pass through the filter.
*/

/* 1D vector of query and training keypoints.
**
**    TODO: matches size < 4 cannot compute homography matrix... handle this!
**    In:   matches, queryKeypoints, trainingKeypoints
**    Out:  matches, homography
*/
void ransacFilter(std::vector<DMatch> &matches, std::vector<KeyPoint> &queryKeypoints, std::vector<KeyPoint> &trainingKeypoints, Mat &homography)
{
  std::vector<Point2f> queryCoords;
  std::vector<Point2f> trainingCoords;

  for(int i = 0; i < matches.size(); i++)
  {
    //Get the coords of the keypoints from the matches
    queryCoords.push_back(queryKeypoints.at(matches.at(i).queryIdx).pt);
    trainingCoords.push_back(trainingKeypoints.at(matches.at(i).trainIdx).pt);
  }

  Mat outputMask;
  homography = findHomography(queryCoords, trainingCoords, CV_RANSAC, 3, outputMask);
  int inlierCounter = 0;
  std::vector<DMatch> good_matches;
  for(int i = 0; i < outputMask.rows; i++)
  {
    if((unsigned int)outputMask.at<uchar>(i))
    {
      inlierCounter++;
      good_matches.push_back(matches[i]);
    }
  }
  //printf("Inliers/Outliers = %d/%d, i.e. %fpc\n", inlierCounter, outputMask.rows, (((float)inlierCounter / (float)outputMask.rows * 100)));
  matches = good_matches;
}

/* 1D vector of query keypoints, a set of training keypoint vectors,
** each usually corresponding to its own training image.
** If no matches are found for a set of training keypoints, the identity
** matrix is pushed onto the homographies list.
**
**    In:   matches, queryKeypoints, trainingKeypoints
**    Out:  matches, homographies
*/
void ransacFilter(std::vector<DMatch> &matches, std::vector<KeyPoint> &queryKeypoints, std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &homographies)
{
  std::vector<Point2f> queryCoords;
  std::vector<Point2f> trainingCoords;
  std::vector<DMatch> good_matches;

  for(int j = 0; j < trainingKeypoints.size(); j++)
  {
    queryCoords.clear();
    trainingCoords.clear();

    for( int i = 0; i < matches.size(); i++ )
    {
      //Get the coords of the keypoints from the matches, making sure to
      //only consider matches derived from keypoints in this training image
      if(matches.at(i).imgIdx == j)
      {
        queryCoords.push_back( queryKeypoints.at(matches.at(i).queryIdx).pt );
        trainingCoords.push_back( trainingKeypoints.at(j).at(matches.at(i).trainIdx).pt );
      }
    }

    Mat outputMask;
    int inlierCounter = 0;

    if(queryCoords.size() != 0 && queryCoords.size() != 0)
    {
      homographies.push_back(findHomography(queryCoords, trainingCoords, CV_RANSAC, 3, outputMask));
    } else {
      //cant compute homography is there were no matches in this training image, so push identity matrix
      homographies.push_back(Mat::eye(3, 3, CV_64F));
    }

    //Filter matches according to RANSAC inliers
    for (int i = 0; i < outputMask.rows; i++) {
      if((unsigned int)outputMask.at<uchar>(i)) {
        inlierCounter++;
        good_matches.push_back(matches[i]);
      }
    }
    printf("Inliers/Outliers = %d/%d, i.e. %fpc\n", inlierCounter, outputMask.rows, (((float)inlierCounter / (float)outputMask.rows * 100)));
  }
  matches = good_matches;
}

/*  Filter a list of query keypoints + descriptors such that each keypoint must
**  be in at least <threshold> matches across all training images.
**
**    In:   queryKeypoints, queryDescriptors, matchesSet, threshold
**    Out:  queryKeypoints, queryDescriptors
*/
void findGoodFeatures(std::vector<KeyPoint> &queryKeypoints, Mat &queryDescriptors, std::vector<std::vector<DMatch> > &matchesSet, int threshold)
{
  //Entry i of countArray corresponds to keypoint i, and counts the number of matches that keypoint is in.
  int* countArray = (int*) calloc(queryKeypoints.size(), sizeof(int));
  for(int i = 0; i < matchesSet.size(); i++)  //for each set of matches (1 per training image)
  {
    for(int j = 0; j < matchesSet.at(i).size(); j++)  //for each match in this set
    {
      countArray[matchesSet.at(i).at(j).queryIdx]++;
    }
  }

  std::vector<KeyPoint> goodKeypoints;
  for(int i = 0; i < queryKeypoints.size(); i++)
  {
    if(countArray[i] >= threshold)
    {
      goodKeypoints.push_back(queryKeypoints.at(i));
    }
  }

  Mat goodDescriptors;
  goodDescriptors.create(goodKeypoints.size(), queryDescriptors.cols, queryDescriptors.type());
  int nextEmptyRow = 0;
  for(int i = 0; i < queryKeypoints.size(); i++)
  {
    if(countArray[i] >= threshold)
    {
      queryDescriptors.row(i).copyTo(goodDescriptors.row(nextEmptyRow));
      nextEmptyRow++;
    }
  }

  queryKeypoints = goodKeypoints;
  queryDescriptors = goodDescriptors;
}

/*  -Take each training image in turn, treating it as a query image.
**  -Fully match it with each of the remaining training images.
**  -Build a list of valid keypoints for this image according to the number
**   of matches each keypoint is a part of, (thresholding at a certain number of matches).
**  -Repeat for each training image, being sure to restore the previous query images keypoints to their original values.
**  -Finally, take all the lists of validated keypoints and use these as the training set.
**  -Now, a new query image can be matched to the training images with these valid keypoint sets.
**
**  This turned out to not work very well! Often, too many keypoints were lost, so the new keypoint-rich query image
**  would match to the same "most similar" remaining keypoint in the training image. Perhaps performance will be better
**  for a much larger training set?
**
**    In:   trainingKeypoints, trainingDescriptors
**    Out:  goodTrainingKeypoints, goodTrainingDescriptors
*/

/*
void findGoodTrainingFeatures(std::vector<std::vector<KeyPoint> > &trainingKeypoints, std::vector<Mat> &trainingDescriptors, std::vector<std::vector<KeyPoint> > &goodTrainingKeypoints, std::vector<Mat> &goodTrainingDescriptors)
{
  std::vector<KeyPoint> currentGoodKeypoints;
  Mat currentGoodDescriptors;

  //For each training image there is a k-length list of match sets, ranked in order of distance
  std::vector<std::vector<std::vector<DMatch> > > knnMatches;
  //For each training image there is a set of matches
  std::vector<std::vector<DMatch> > matchesSet;
  //For a single image there is a set of matches
  std::vector<DMatch> matches;

  std::vector<char> mask;
  mask.resize(trainingKeypoints.size());

  for(int i = 0; i < trainingKeypoints.size(); i++)
  {
    //mask the query image in the training set
    fill(mask.begin(), mask.end(), 0);
    mask.at(i) = 1;

    matches.clear();
    knnMatches.clear();
    matchesSet.clear();

    //Get the kNN matches for the query image against each training image
    matchKnn(trainingDescriptors.at(i), trainingDescriptors, knnMatches, 2, mask);

    //Filter each kNN match set and produce a list of best matches for the query
    //image against each training image, (matchesSet).
    for(int k = 0; k < knnMatches.size(); k++)
    {
      int trainingIndex = (k >= i) ? k + 1 : k;
      loweFilter(knnMatches.at(k), matches);
      Mat homography;
      ransacFilter(matches, trainingKeypoints.at(i), trainingKeypoints.at(trainingIndex), homography);
      matchesSet.push_back(matches);
    }

    Mat out;
    currentGoodKeypoints = trainingKeypoints.at(i);
    currentGoodDescriptors = trainingDescriptors.at(i);

    findGoodFeatures(currentGoodKeypoints, currentGoodDescriptors, matchesSet, 2);

    goodTrainingKeypoints.push_back(currentGoodKeypoints);
    goodTrainingDescriptors.push_back(currentGoodDescriptors);
  }

}*/

/* Consider the bounding box of input image as the object.
** Tranform this box according to the homography matrix and draw it on the
** output image.
*/
void drawObject(Mat &input, Mat &homography, Mat &output)
{
  std::vector<Point2f> objCorners(4);
  objCorners[0] = Point(0,0);
  objCorners[1] = Point( input.cols, 0 );
  objCorners[2] = Point( input.cols, input.rows );
  objCorners[3] = Point( 0, input.rows );

  std::vector<Point2f> scnCorners(4);

  perspectiveTransform(objCorners, scnCorners, homography);

  line( output, scnCorners[0] + Point2f( input.cols, 0), scnCorners[1] + Point2f( input.cols, 0), Scalar(0, 255, 0), 4);
  line( output, scnCorners[1] + Point2f( input.cols, 0), scnCorners[2] + Point2f( input.cols, 0), Scalar( 0, 255, 0), 4);
  line( output, scnCorners[2] + Point2f( input.cols, 0), scnCorners[3] + Point2f( input.cols, 0), Scalar( 0, 255, 0), 4);
  line( output, scnCorners[3] + Point2f( input.cols, 0), scnCorners[0] + Point2f( input.cols, 0), Scalar( 0, 255, 0), 4);
}
