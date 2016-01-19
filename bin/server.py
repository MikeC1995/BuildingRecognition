import cv2
import pbcvt
import recogniser

r = recogniser.Recogniser("wills")

image = cv2.imread("/root/server/images/wills/query/0001.jpg")
matches = r.query("/root/server/images/wills/query/0001.jpg")
print matches
