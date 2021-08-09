import numpy as np
import torch
import torch.utils.data
import yaml
from model.model import RegressionModel
from PIL import Image
import torchvision.transforms as T
# Turn off the warning of YAML loader
import warnings

warnings.filterwarnings('ignore')


# CUDA settings
def setup_cuda():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setting seeds (optional)
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return device


# Main
if __name__ == '__main__':
    # 1. Setup CUDA
    device = setup_cuda()

    # 2. Load the configurations from the yaml file
    config_path = './configs/config.yml'
    with open(config_path) as file:
        cfg = yaml.load(file)

    img_size = cfg['train_params']['img_size']
    lr_start = cfg['train_params']['lr_start']
    backbone = cfg['model_params']['backbone']
    max_age = cfg['model_params']['max_age']
    age_group = cfg['model_params']['age_group']
    num_classes = int(max_age / age_group) + 1

    # 3. Specify the trained weights and input image
    weight_path = './checkpoints/epoch_120_mae_6.8016.pt'
    img_path = '/data/age_estimation/dataset/UTKFace/aligned-cropped/UTKFace/36_1_0_20170104234532883.jpg.chip.jpg'

    # 4. Load the trained model
    model = RegressionModel(num_classes, backbone=backbone).to(device)
    model.load_state_dict(torch.load(weight_path, device))
    model = model.to(device)
    print('Loading the trained model done')

    # 5. Process the input image
    img = Image.open(img_path)
    transforms = T.Compose([
        T.Resize((img_size, img_size), interpolation=Image.BILINEAR),
        T.ToTensor()
    ])
    img = transforms(img)
    img = img.unsqueeze(0).to(device)  # expand the tensor along the first dimension (i.e. batch size of 1)

    # 6. Predict age
    _, probas = model(img)
    pred_level = probas > 0.5
    pred_age = torch.sum(pred_level, dim=1)
    # img.show()
    print('Predicted age: ', pred_age.item())
