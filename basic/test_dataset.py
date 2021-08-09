"""
UOW, Wed Feb 24 23:37:42 2021
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import yaml
from model.model import RegressionModel
# Turn off the warning of YAML loader
import warnings

warnings.filterwarnings('ignore')


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


# Test the model
def test_model(model, device, test_loader):
    model.eval()
    mae, mse = 0, 0
    pbar = tqdm(test_loader, ncols=80)

    with torch.no_grad():
        for iteration, (img, true_age, _) in enumerate(pbar):
            img = img.to(device)
            true_age = true_age.to(device, dtype=torch.float)

            # Perform a forward pass
            pred_age = model(img)

            # Calculate the current performance
            num_imgs += true_age.size(0)
            mae += torch.sum(torch.abs(pred_age - true_age))
            mse += torch.sum((pred_age - true_age) ** 2)

    mae = mae / num_imgs
    mse = mse / num_imgs

    return mae, mse


# Main
if __name__ == '__main__':
    # 1. Setup CUDA
    device = setup_cuda()

    # 2. Load the configurations from the yaml file
    config_path = './configs/config.yml'
    with open(config_path) as file:
        cfg = yaml.load(file)

    dataset_dir = cfg['train_params']['dataset_dir']
    test_list = cfg['train_params']['test_list']
    train_list = cfg['train_params']['train_list']
    valid_list = cfg['train_params']['valid_list']
    weight_path = cfg['train_params']['weight_path']
    num_workers = cfg['train_params']['num_workers']
    img_size = cfg['train_params']['img_size']
    max_age = cfg['model_params']['max_age']
    age_group = cfg['model_params']['age_group']
    num_classes = int(max_age / age_group) + 1

    # 3. Load the dataset
    from utils.load_dataset import FaceDataset

    test_dataset = FaceDataset(dataset_dir=dataset_dir,
                               data_list_file=test_list,
                               age_group=age_group,
                               num_classes=num_classes,
                               input_shape=(img_size, img_size))
    test_loader = DataLoader(test_dataset,
                             batch_size=1,  # must be 1 for testing phase
                             shuffle=True,
                             num_workers=num_workers)
    train_dataset = FaceDataset(dataset_dir=dataset_dir,
                                data_list_file=train_list,
                                age_group=age_group,
                                num_classes=num_classes,
                                input_shape=(img_size, img_size))
    train_loader = DataLoader(train_dataset,
                              batch_size=1,  # must be 1 for testing phase
                              shuffle=True,
                              num_workers=num_workers)
    valid_dataset = FaceDataset(dataset_dir=dataset_dir,
                                data_list_file=valid_list,
                                age_group=age_group,
                                num_classes=num_classes,
                                input_shape=(img_size, img_size))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,  # must be 1 for testing phase
                              shuffle=True,
                              num_workers=num_workers)

    print('#Train images: {}, #Test images: {}, #Valid images: {}'.format(train_dataset.__len__(),
                                                                          test_dataset.__len__(),
                                                                          valid_dataset.__len__()))

    print('Age group: {}. Number of classes: {}'.format(age_group, num_classes))
    print('Train list: {}. Test list: {}. Valid list: {}'.format(train_list, test_list, valid_list))
    print('Trained network: ', cfg['train_params']['weight_path'])

    # 4. Load the trained model
    model = SSR_Net(stage_num=[3, 3, 3], lambda_local=1, lambda_d=1, max_age=max_age).to(device)
    model.load_state_dict(torch.load(weight_path))
    print('Loading the trained model done')
    # Check Keras-like model summary
    summary(model, input_size=(3, img_size, img_size))

    # 6. Start validate the model
    mae, mse = test_model(model, device, train_loader)
    print('Training MAE: {}, Training MSE: {}'.format(mae, mse))

    mae, mse = test_model(model, device, test_loader)
    print('Test MAE: {}, Test MSE: {}'.format(mae, mse))

    mae, mse = test_model(model, device, valid_loader)
    print('Validation MAE: {}, Validation MSE: {}'.format(mae, mse))
