"""
UOW, Mon Dec 28 14:37:42 2020
Dependencies: torch>=1.1, torchvision>=0.3.0
"""
from PIL import Image, ImageDraw
from dsfd.detect import DSFDDetector
from timeit import default_timer as timer
import numpy as np
import argparse


def dsfd(in_img, out_img, conf=0.5, iou=0.3):
    # Load the pre-trained DSFD
    weight_path = "./weights/WIDERFace_DSFD_RES152.pth"
    detector = DSFDDetector(weight_path)

    # Read the input image
    img = Image.open(in_img)

    # Perform face detection
    start = timer()
    detections = detector.detect_face(np.asarray(img),
                                      confidence_threshold=conf,
                                      nms_iou_threshold=iou)
    end = timer()
    num_faces = detections.shape[0]
    print('Total detected faces: {}. Detection time: {}'.format(num_faces, end - start))

    # Visualize the results
    draw = ImageDraw.Draw(img)
    for k in range(num_faces):
        draw.rectangle(((detections[k][0], detections[k][1]), (detections[k][2], detections[k][3])),
                       outline='magenta',
                       width=3)

    # Save the output image
    img.save(out_img)
    img.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Runtime script for face detection')
    args.add_argument('--input', default=None, type=str, help='File path of input image.')
    args.add_argument('--output', default=None, type=str, help='File path of output image.')
    args.add_argument('--conf', default=0.5, type=float, help='Confidence threshold for removing weak detections.')
    args.add_argument('--iou', default=0.3, type=float, help='IOU threshold for removing duplicate detections.')
    cmd_args = args.parse_args()
    assert cmd_args.input is not None, "Specify the file path of input image"
    assert cmd_args.output is not None, "Specify the file path of output image"

    dsfd(in_img=cmd_args.input,
         out_img=cmd_args.output,
         conf=cmd_args.conf,
         iou=cmd_args.iou)
