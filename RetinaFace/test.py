"""
UOW, Mon Dec 28 14:37:42 2020
"""
import torch
from PIL import Image, ImageDraw
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from timeit import default_timer as timer
from utils.utils import load_model


def retinaface(in_img, out_img, backbone='mobilenet', conf_thres=0.5,
               iou_thres=0.3, top_k=5000, keep_top_k=750):
    # Load the configuration
    if backbone == "mobilenet":
        trained_model = './weights/mobilenet0.25_Final.pth'
        cfg = cfg_mnet
    elif backbone == "resnet50":
        trained_model = './weights/Resnet50_Final.pth'
        cfg = cfg_re50

    # Load the trained RetinaFace model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model)
    net.eval()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # Read and pre-process the input image
    img_raw = Image.open(in_img)
    img = np.float32(np.asarray(img_raw))
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
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
    landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landmarks = landmarks * scale1
    landmarks = landmarks.cpu().numpy()

    # Ignore weak detections with low confidence scores
    inds = np.where(scores > conf_thres)[0]
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # Keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    # Do NMS
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(detections, iou_thres)
    detections = detections[keep, :]
    landmarks = landmarks[keep]

    # Keep top-K faster NMS
    detections = detections[:keep_top_k, :]
    landmarks = landmarks[:keep_top_k, :]

    # (num_faces, 15): 
    #      (0, 1, 2, 3): bounding box
    #      4: visual score
    #      (5, 6), (7, 8), (9, 10), (11, 12), (13, 14): landmarks
    detections = np.concatenate((detections, landmarks), axis=1)

    #  Visualize the results
    draw = ImageDraw.Draw(img_raw)
    for det in detections:
        draw.rectangle(((det[0], det[1]), (det[2], det[3])),
                       outline='cyan',
                       width=3)

    print('Total detected faces: {}. Detection time: {}'.format(detections.shape[0], end - start))

    # Save the output image
    img_raw.save(out_img)
    img_raw.show()


if __name__ == '__main__':
    retinaface(in_img='test4.jpg',
               out_img='resnet_test4.jpg',
               backbone='resnet50')
    retinaface(in_img='test4.jpg',
               out_img='resnet_test4.jpg',
               backbone='resnet50')
    retinaface(in_img='test4.jpg',
               out_img='resnet_test4.jpg',
               backbone='resnet50')
