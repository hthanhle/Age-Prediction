"""
UOW, Wed Feb 24 23:37:42 2021
Dependencies: torch>=1.1, torchvision>=0.3.0
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import Adam
import yaml
from network.SSR_Net import SSR_Net
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


# Train the model over a single epoch
def train_model(model, device, optimizer, loss_fn, train_loader):
    model.train()
    mae, mse, num_imgs, train_loss = 0, 0, 0, 0

    pbar = tqdm(train_loader, ncols=80, desc='Training')

    for iteration, (img, true_age, _) in enumerate(pbar):
        img = img.to(device)
        true_age = true_age.to(device, dtype=torch.float)

        # Perform a forward pass
        pred_age = model(img)

        # Calculate the training loss
        loss = loss_fn(pred_age, true_age)
        train_loss += loss.item()
        # print('\n---------------------------')
        # print('Pred age: ', pred_age)
        # print('True age: ', true_age)
        # print('loss: ', loss)

        # Calculate the current performance
        num_imgs += true_age.size(0)
        mae += torch.sum(torch.abs(pred_age - true_age))
        mse += torch.sum((pred_age - true_age) ** 2)
        # print('MSE: ', mse)
        # print('MAE: ', mae)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

    mae = mae/num_imgs
    mse = mse/num_imgs

    return train_loss / len(train_loader), mae, mse


# Validate the model over a single epoch
def validate_model(model, device, loss_fn, valid_loader):
    model.eval()
    mae, mse, num_imgs, valid_loss = 0, 0, 0, 0
    with torch.no_grad():
        for iteration, (img, true_age, _) in enumerate(valid_loader):
            img = img.to(device)
            true_age = true_age.to(device, dtype=torch.float)

            # Perform a forward pass
            pred_age = model(img)

            # Calculate the training loss
            loss = loss_fn(pred_age, true_age)
            valid_loss += loss.item()

            # Calculate the current performance
            num_imgs += true_age.size(0)
            mae += torch.sum(torch.abs(pred_age - true_age))
            mse += torch.sum((pred_age - true_age) ** 2)

    mae = mae/num_imgs
    mse = mse/num_imgs

    return valid_loss / len(valid_loader), mae, mse


# Main
if __name__ == '__main__':

    # 1. Setup CUDA
    device = setup_cuda()

    # 2. Load the configurations from the yaml file
    config_path = './configs/config.yml'
    with open(config_path) as file:
        cfg = yaml.load(file)

    dataset_dir = cfg['train_params']['dataset_dir']
    train_list = cfg['train_params']['train_list']
    test_list = cfg['train_params']['test_list']
    valid_list = cfg['train_params']['valid_list']
    num_epochs = cfg['train_params']['num_epochs']
    batch_size = cfg['train_params']['batch_size']
    num_workers = cfg['train_params']['num_workers']
    img_size = cfg['train_params']['img_size']
    lr_start = cfg['train_params']['lr_start']
    max_age = cfg['model_params']['max_age']
    age_group = cfg['model_params']['age_group']
    num_classes = int(max_age / age_group) + 1

    # 3. Load the dataset
    from utils.load_dataset import FaceDataset

    train_dataset = FaceDataset(dataset_dir=dataset_dir,
                                data_list_file=train_list,
                                age_group=age_group,
                                num_classes=num_classes,
                                input_shape=(img_size, img_size))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    valid_dataset = FaceDataset(dataset_dir=dataset_dir,
                                data_list_file=valid_list,
                                age_group=age_group,
                                num_classes=num_classes,
                                input_shape=(img_size, img_size))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    print('Number of training images: {}. Number of validation images: {}'.format(train_dataset.__len__(),
                                                                                  valid_dataset.__len__()))
    print('Age group: {}. Number of classes: {}'.format(age_group, num_classes))

    # 4. Create a SSRNet model
    model = SSR_Net(stage_num=[3, 3, 3], lambda_local=1, lambda_d=1, max_age=max_age).to(device)
    if cfg['train_params']['resumed']:
        model.load_state_dict(torch.load(cfg['train_params']['weight_path'], device))
        print('Loading the saved checkpoint done. Training process resumed...')

    # 5. Specify loss function and optimizer
    optimizer = Adam(model.parameters(), lr=lr_start)
    loss_fn = torch.nn.L1Loss()  # MAE loss

    # 6. Start training the model
    best_mae = 999
    for epoch in range(num_epochs):
        # 6.1. Train the model over a single epoch
        train_loss, train_mae, train_mse = train_model(model, device, optimizer, loss_fn, train_loader)

        # 6.2. Validate the model
        valid_loss, valid_mae, valid_mse = validate_model(model, device, loss_fn, valid_loader)

        print('Epoch: {} \tTraining MAE: {:.4f} \tValid MAE: {:.4f} \tTraining MSE: {:.4f} \tValid MSE: {:.4f}'.format(
            epoch, train_mae, valid_mae, train_mse, valid_mse))

        # 6.3. Save the model if the validation performance is increasing
        if best_mae >= valid_mae:
            print('Valid MAE decreased ({:.4f} --> {:.4f}). Model saved'.format(best_mae, valid_mae))
            torch.save(model.state_dict(),
                       './checkpoints/epoch_' + str(epoch) + '_mae_{0:.4f}'.format(valid_mae) + '.pt')
            best_mae = valid_mae
