import numpy as np
import torch
import torch.utils.data
from PIL import Image
from model.model import get_model
import yaml
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

    dataset_dir = cfg['train_params']['dataset_dir']
    img_size = cfg['train_params']['img_size']
    backbone = cfg['model_params']['backbone']
    num_ages = cfg['model_params']['num_ages']
    img_size = cfg['train_params']['img_size']

    # 3. Specify the trained weights and input image
    weight_path = './checkpoints/.....'
    img_path = '.....'

    # 4. Load the trained model
    model = get_model(model_name=backbone, num_ages=num_ages)
    model.load_state_dict(torch.load(weight_path, device))
    model = model.to(device)
    print('Loading the trained model done')

    # 5. Predict age
    img = Image.open(img_path)
    transforms = T.Compose([
        T.Resize((img_size, img_size), interpolation=Image.BILINEAR),
        T.ToTensor()
    ])
    img = transforms(img)
    img = img.unsqueeze(0).to(device)  # expand the tensor along the first dimension (i.e. batch size of 1)

    outputs = model(img)
    _, pred_age = outputs.max(1)

    img.show()
    print('Predicted age: ', pred_age.item())
