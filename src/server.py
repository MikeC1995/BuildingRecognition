import cv2
import saveable_matcher

matcher = saveable_matcher.SaveableFlannBasedMatcher("wills")
matcher.test()
matcher.load()
print "Loaded matcher!"

image = cv2.imread("/root/server/images/wills/query/0001.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

# Methods below cause OutOfMemoryError. Instead, wrap surf.cpp modules
# for python. Need to be able to:
#   -Detect and compute kps and descs for query image
#   -call matcher->knnMatch on loaded matcher
#   -call loweFilter in surf module

(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
