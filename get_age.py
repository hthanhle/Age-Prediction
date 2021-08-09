import argparse
from PIL import ImageDraw, Image, ImageFont
import torch
import numpy as np
import torchvision.transforms as T
from RetinaFace.retinaface import retinaface
from SSRNet.network.SSR_Net import SSR_Net
from DSFD import dsfd
from CORAL.model.model import RegressionModel
from timeit import default_timer as timer
import cv2
from RetinaFace.data import cfg_mnet, cfg_re50
from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.utils.utils import load_model
from DSFD.dsfd.detect import DSFDDetector


# CUDA settings
def setup_cuda():
    # Setting seeds (optional)
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Runtime script for age estimation')
    args.add_argument('-i', '--input', default=None, type=str, help='File path of input image.')
    args.add_argument('-o', '--output', default=None, type=str, help='File path of output image.')
    args.add_argument('-d', '--detector', default='retinaface', type=str, help='Detector name: retinaface/dsfd')
    args.add_argument('-e', '--estimator', default='ssrnet', type=str, help='Age estimator name: coral/ssrnet')
    cmd_args = args.parse_args()
    assert cmd_args.input is not None, "Specify the file path of input image"
    assert cmd_args.output is not None, "Specify the file path of output image"

    # 2. Setup CUDA
    device = setup_cuda()

    # 3. Read an input image
    img = cv2.imread(cmd_args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. Load the trained face detector (i.e., Stage 1)
    if cmd_args.detector == 'retinaface':
        retina = RetinaFace(cfg=cfg_re50, phase='test')
        retina = load_model(retina, './RetinaFace/weights/Resnet50_Final.pth')
        retina.eval()
        retina = retina.to(device)
        print('Loading the trained RetinaFace done')
    else:
        assert cmd_args.detector == 'dsfd', 'Detector name must be either \'retinaface\' or \'dsfd\''
        dsfd_model = DSFDDetector('./DSFD/weights/WIDERFace_DSFD_RES152.pth')
        print('Loading the trained DSFD done')

    # 5. Load the trained estimator (i.e., Stage 2)
    if cmd_args.estimator == 'ssrnet':
        estimator = SSR_Net(stage_num=[3, 3, 3], lambda_local=1, lambda_d=1, max_age=116).to(device)
        estimator.load_state_dict(torch.load('./SSRNet/checkpoints/epoch_8_mae_5.9158.pt'),
                                  strict=False)  # ignore no-matching keys
        print('Loading the trained SSRNet done')
    else:
        assert cmd_args.detector == 'coral', 'Estimator name must be either \'ssrnet\' or \'coral\''
        estimator = RegressionModel(101, backbone='resnext101_32x8d').to(device)
        estimator.load_state_dict(torch.load('./CORAL/checkpoints/epoch_2_mae_10.8020.pt', device))
        estimator = estimator.to(device)
        print('Loading the trained CORAL done')

    # 6. (Stage 1) Perform face detection
    start = timer()
    if cmd_args.detector == 'retinaface':
        detections = retinaface(in_img=img, net=retina, device=device, cfg=cfg_re50)
    else:
        detections = dsfd(model=dsfd_model, in_img=img)
    end = timer()
    print('Number of detected faces: {}. Detection time: {}'.format(detections.shape[0], end - start))

    # 7. Specify transformations applied to the cropped images (i.e. detected faces)
    transforms = T.Compose([
        T.Resize((64, 64), interpolation=Image.BILINEAR),  # resize
        T.ToTensor()  # then convert to a Pytorch tensor
    ])

    # 8. Process the detected faces
    font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', size=20)  # for a better visualization
    img = Image.fromarray(img)  # convert to a PIL image for visualization

    for det in detections:
        # 8.1. Pre-process the bounding box
        top = max(0, np.floor(det[1] + 0.5).astype('int32'))
        left = max(0, np.floor(det[0] + 0.5).astype('int32'))
        bottom = min(img.size[1], np.floor(det[3] + 0.5).astype('int32'))
        right = min(img.size[0], np.floor(det[2] + 0.5).astype('int32'))

        # 8.2. Crop the detected face from the input PIL image
        face = img.crop((top, left, bottom, right))

        # 8.3. Apply the transformations
        face = transforms(face)
        face = face.unsqueeze(0).to(device)  # expand the tensor along the first dimension (i.e. batch size of 1)

        # 8.4. Estimate age
        start = timer()
        if cmd_args.estimator == 'ssrnet':
            pred_age = estimator(face)
        else:  # 'coral'
            _, prob = estimator(face)
            pred_level = prob > 0.5
            pred_age = torch.sum(pred_level, dim=1)

        pred_age = pred_age.item()  # get a number from the Pytorch tensor 'pred_age' containing a single value
        end = timer()
        print('Age estimation time: ', end - start)

        # 8.5. Visualize the result
        draw = ImageDraw.Draw(img)
        label = '{:.1f}'.format(pred_age)
        label_size = draw.textsize(label, font)

        # Adjust the origin if the text is outside
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])  # display the text as normal (on the top of rectangle)
        else:
            text_origin = np.array([left, top + 1])  # display the text inside the rectangle)

        # Draw a rectangle to cover the detected face
        draw.rectangle(((left, top), (right, bottom)), outline='cyan', width=3)

        # Draw an outside rectangle to cover the text
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='cyan')

        # Draw the text
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    # Save the output image
    img.save(cmd_args.output)
    img.show()
