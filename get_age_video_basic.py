import argparse
from PIL import ImageDraw, Image, ImageFont
import torch
import numpy as np
import torchvision.transforms as T
from RetinaFace.retinaface import retinaface
from RetinaFace.data import cfg_mnet, cfg_re50
from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.utils.utils import load_model
from basic.model.model import RegressionModel
from timeit import default_timer as timer
from utils.utils import check_duplicate
import cv2


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
    args.add_argument('-i', '--input', default=None, type=str, help='File path of input video.')
    args.add_argument('-o', '--output', default=None, type=str, help='File path of output video.')
    cmd_args = args.parse_args()
    assert cmd_args.input is not None, "Specify the file path of input image"
    assert cmd_args.output is not None, "Specify the file path of output image"

    # 2. Setup CUDA
    device = setup_cuda()

    # 3. Load the trained RetinaFace for face detection (i.e., Stage 1)
    retina = RetinaFace(cfg=cfg_re50, phase='test')
    retina = load_model(retina, './RetinaFace/weights/Resnet50_Final.pth')
    retina.eval()
    retina = retina.to(device)
    print('Loading ResNet-50 RetinaFace done')

    # 4. Load the trained SSRNet for age estimation (i.e., Stage 2)
    model = RegressionModel(num_classes=117, backbone='mobilenet_v2').to(device)
    model.eval()
    model.load_state_dict(torch.load('./basic/checkpoints/epoch_78_mae_5.2784.pt'),
                          strict=False)  # ignore no-matching keys
    print('Loading the Basic Regression model done')

    # 5. Capture the input video
    in_video = cv2.VideoCapture(cmd_args.input)
    frame_width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Original resolution: ', frame_width, ', ', frame_width)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter(cmd_args.output, fourcc, 24, (frame_width, frame_height))

    pre_detections = []
    pre_ages = []
    font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', size=20)  # font used for visualization
    # Specify transformations applied to the detected faces (i.e., cropped images)
    transforms = T.Compose([
        T.Resize((64, 64), interpolation=Image.BILINEAR),  # resize
        T.ToTensor()  # then convert to a Pytorch tensor (normalization also included)
    ])

    while True:
        # 3.1. Read a frame from the video
        result, frame = in_video.read()
        if not result:  # if the video finished
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3.2. (Stage 1) Perform face detection using RetinaFace
        start = timer()
        detections = retinaface(in_img=frame, net=retina, device=device, cfg=cfg_re50)
        detections = detections[:, : 4]  # get the bounding boxes
        end = timer()
        print('Number of detected faces: {}. Detection time: {}'.format(detections.shape[0], end - start))

        # 3.5. Process the detected faces
        img = Image.fromarray(frame)  # convert to a PIL image for visualization

        cur_idx = []  # a list keeps the indices of the detections that are present in the current frame

        for det in detections:

            # 6.1. Pre-process the bounding box
            top = max(0, np.floor(det[1] + 0.5).astype('int32'))
            left = max(0, np.floor(det[0] + 0.5).astype('int32'))
            bottom = min(img.size[1], np.floor(det[3] + 0.5).astype('int32'))
            right = min(img.size[0], np.floor(det[2] + 0.5).astype('int32'))

            # 6.2. Crop the detected face from the input PIL image
            face = img.crop((top, left, bottom, right))

            # 6.3. Apply the transformations
            face = transforms(face)
            face = face.unsqueeze(0).to(device)  # expand the tensor along the first dimension (i.e. batch size of 1)

            # 6.4. Perform age estimation
            prob = model(face)
            pred_level = prob > 0.5
            pred_age = torch.sum(pred_level, dim=1).item()

            # 6.5. Check duplicate bounding boxes over frames
            idx = check_duplicate(det, pre_detections)
            if idx is None:  # if the current detection is a new one (not exist in the list)
                pre_detections.append(
                    det)  # 'pre_detections' is a list whose each element is a Numpy array of size (1, 4) as box coordinates
                pre_ages.append([
                                    pred_age])  # 'pre_ages' is a list whose each element is also a list for storing all previous predicted ages
                cur_idx.append(len(pre_detections) - 1)  # store the index of the current detection

            else:  # otherwise, we update the entries in the lists
                pre_detections[idx] = det
                pre_ages[idx].append(pred_age)
                pred_age = np.mean(pre_ages[idx])  # average all previous predictions
                cur_idx.append(idx)  # store the index of the current detection
                # print('Average prediction age: ', pred_age)

            # 6.6. Visualize the result
            draw = ImageDraw.Draw(img)
            label = '{:.1f}'.format(pred_age)
            label_size = draw.textsize(label, font)

            # Adjust the origin if the text is outside
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # Draw a rectangle to cover the detected face
            draw.rectangle(((left, top), (right, bottom)), outline='cyan', width=3)

            # Draw an outside rectangle to cover the text
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='cyan')

            # Draw the text
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        # Only keep predictions in the list that are present in the current frame
        pre_detections = [pre_detections[k] for k in range(len(pre_detections))
                          if k in cur_idx]
        pre_ages = [pre_ages[k] for k in range(len(pre_ages))
                    if k in cur_idx]

        # 3.6. Write the output frame
        out_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out_video.write(out_frame)
        cv2.imshow("Input video", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow("Machine video", out_frame)

        # 3.7. Early break the loop if ESC is pressed
        if cv2.waitKey(1) & 0xff == 27:
            in_video.release()
            out_video.release()
            cv2.destroyAllWindows()
            break

    in_video.release()
    out_video.release()
    cv2.destroyAllWindows()
