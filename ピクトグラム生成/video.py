import os
import math
import glob
import sys
import cv2 as cv

pathall = os.path.dirname(os.path.abspath(__file__))

width = sys.argv[1]
height = sys.argv[2]
video_fps = sys.argv[3]


output = str(pathall) + '/ou_vi/pictogram.mp4'
# encoder(for mp4)
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv.VideoWriter(output,fourcc, float(video_fps), (int(width), int(height)))

if not video.isOpened():
    print("can't be opened1")

n_mp4 = 0
while True :
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv.imread(str(pathall) + '/image_second/' + str(n_mp4) + '.jpg')

    # can't read image, escape
    if img is None:
        print("can't read2")
        break

    # add
    video.write(img)

    n_mp4 += 1

video.release()
print("success")
