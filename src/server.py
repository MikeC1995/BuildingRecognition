import cv2
import recogniser

r = recogniser.Recogniser("wills")
r.test()

image = cv2.imread("/root/server/images/wills/query/0001.jpg")
r.query(image)

# Methods below cause OutOfMemoryError. Instead, wrap surf.cpp modules
# for python. Need to be able to:
#   -Detect and compute kps and descs for query image
#   -call matcher->knnMatch on loaded matcher
#   -call loweFilter in surf module


# Strategy:
# Do ALL computation in C++. Load matcher into python, and pass matcher
# as argument to C++ function along with image, which returns the classification
# result as a string!
# I.e. rewrite recogniser.cpp to have single query method, accepting image and matcher,
# and returning classification result. Wrap this function with Boost.

#(kps, descs) = sift.detectAndCompute(gray, None)
#print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
