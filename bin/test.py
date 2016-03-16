import cv2
import os
import data_generator

dg = data_generator.DataGenerator()
dg.bigTree("sv/filenames.txt", "sv/features")
