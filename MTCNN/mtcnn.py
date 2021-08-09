'''
UOW, 24/12/2020
Dependencies: keras>=2.0.0
'''
import argparse
from PIL import Image, ImageDraw
from mtcnn import MTCNN
import numpy as np
from timeit import default_timer as timer
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)


def mtcnn(in_img, out_img):
    # Create an MTCNN model
    detector = MTCNN()

    # Read an input image
    img = Image.open(in_img)

    # Perform face detection
    start = timer()
    result = detector.detect_faces(np.asarray(img))
    end = timer()
    print('Total detected faces: {}. Detection time: {}'.format(len(result), end - start))

    # Visualize the result
    draw = ImageDraw.Draw(img)

    for k in range(len(result)):  # process each detected face
        box = result[k]['box']
        # keypoints = result[k]['keypoints']

        # Draw the bounding box
        draw.rectangle(((box[0], box[1]), (box[0] + box[2], box[1] + box[3])),
                       outline='greenyellow',
                       width=3)

        # Draw the key points
        # cv2.circle(image,(keypoints['left_eye']), 2, (0,255,0), 1)
        # cv2.circle(image,(keypoints['right_eye']), 2, (0,255,0), 1)
        # cv2.circle(image,(keypoints['nose']), 2, (255,0,0), 1)
        # cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,155), 1)
        # cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,155), 1)
    # Save the output image
    img.save(out_img)
    img.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Runtime script for face detection')
    args.add_argument('--input', default=None, type=str, help='File path of input image.')
    args.add_argument('--output', default=None, type=str, help='File path of output image.')
    cmd_args = args.parse_args()
    assert cmd_args.input is not None, "Specify the file path of input image"
    assert cmd_args.output is not None, "Specify the file path of output image"

    mtcnn(in_img=cmd_args.input,
          out_img=cmd_args.output)
