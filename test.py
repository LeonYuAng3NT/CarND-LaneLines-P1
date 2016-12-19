
import os
os.listdir()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# Import everything needed to edit/save/watch video clips
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
imageio.plugins.ffmpeg.download()
def process_image(image):
    # The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    #Read in image
    #Grayscale the image
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #Define a kernel size and apply Gaussian smoothing (blurring)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

    #Define the parameters for the Canny edge detection algorithm and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #Create a "masked edges" image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    #Define a four-side polygon to mask the image
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),(450,290),(490, 290),(imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges,mask)

    #Define the Hough transform parameters
    #Make a blank the same size as our image to draw on
    rho = 2 #Distance resolution in pixels of the Hough grid
    theta = np.pi/180 #Angular resolution in radians of the Hough grid
    threshold = 15 #Minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #Minimum number of pixels making up a line
    max_line_gap = 20 #Maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 #Creating a blank to draw lines on

    #Run Hough on edge detected image
    #Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    #Iterate over the output "lines' and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    #Create a "color" binary image to combine with the line image
    color_edges = np.dstack((edges, edges, edges))

    #Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return lines_edges


yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
