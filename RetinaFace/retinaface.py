"""
UOW, Mon Dec 28 14:37:42 2020
"""
import torch
from PIL import Image, ImageDraw
import numpy as np
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms
from RetinaFace.utils.box_utils import decode
from timeit import default_timer as timer


# For a real-time demo: 1) all initializations are put outside, and 2) landmark detection is removed
def retinaface(in_img, net, device, cfg,
               out_img=None,
               conf_thres=0.5,
               iou_thres=0.3,
               top_k=5000,
               keep_top_k=750):
    """
    :param in_img: input image as a Numpy array of size (H, W, 3)
    :param net: a trained RetinaFace network
    :param device: a device type ('cpu' or 'cuda')
    :param cfg: config for the backbone MobileNet/ResNet50(cfg_mnet or cfg_re50)
    :param out_img: file path of output image
    :param conf_thres: confidence threshold for removing weak detections
    :param iou_thres: IOU threshold for the NMS algorithm
    :param top_k: top k detections before NMS
    :param keep_top_k: top k final detections
    :return: detections as a Numpy array of size (num_faces, 15), where:
            (0, 1, 2, 3): coordinates of bounding boxes x1, y1, x2, y2
            4: visual score
            (5, 6), (7, 8), (9, 10), (11, 12), (13, 14): landmarks. But they are removed in this version.
    """

    # Read and pre-process the input image
    im_height, im_width, _ = in_img.shape
    scale = torch.Tensor([in_img.shape[1], in_img.shape[0], in_img.shape[1], in_img.shape[0]])
    img = in_img - (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device, dtype=torch.float)
    scale = scale.to(device)

    start = timer()
    loc, conf, landmarks = net(img)  # forward pass
    end = timer()

    # Post-process the detections
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().detach().numpy()
    scores = conf.squeeze(0).cpu().detach().numpy()[:, 1]

    # Ignore weak detections with low confidence scores
    idx = np.where(scores > conf_thres)[0]
    boxes = boxes[idx]
    scores = scores[idx]

    # Keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # Do NMS
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(detections, iou_thres)
    detections = detections[keep, :]

    # Keep top-K detections
    detections = detections[:keep_top_k, :]

    #  Visualize the results if output image is specified
    if out_img is not None:
        img_raw = Image.fromarray(in_img)
        draw = ImageDraw.Draw(img_raw)
        for det in detections:
            draw.rectangle(((det[0], det[1]), (det[2], det[3])),
                           outline='cyan',
                           width=3)
        # Save the output image
        img_raw.save(out_img)
        img_raw.show()

    # Otherwise, just return the detections for the next stage
    else:
        return detections
