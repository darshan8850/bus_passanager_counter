import sys
import time
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

PATH_TO_CKPT_HEAD = '/content/Human-Head-Detection/models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'
head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)
image_path = '/content/Human-Head-Detection/frame_0050_faces_4.jpg'

output_filename = 'output2.jpg'

if __name__ == "__main__":
    image = cv2.imread(image_path)  # Read the image using OpenCV
    t_start = time.time()

    im_height, im_width, im_channel = image.shape
    image = cv2.flip(image, 1)

    # Head-detection run model
    image, heads = head_detector.run(image, im_width, im_height)

    fps = 1 / (time.time() - t_start)
    cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), 0, 5e-3 * 130, (0, 0, 255), 2)
    cv2.putText(image, "HEAD DETECTION ", (int(im_width / 2) + 50, im_height - 10), 0, 0.5, (255, 255, 255), 1)

    # Save the image to a file
    cv2.imwrite(output_filename, image)