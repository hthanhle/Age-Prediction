from tqdm import tqdm
import numpy as np
import torch
import os
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import Adam
import yaml
from model.model import RegressionModel
from utils.utils import loss_fn, get_ages, task_importance_weights
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
def train_model(model, device, optimizer, train_loader, imp):
    model.train()
    mae, mse, num_examples, train_loss = 0, 0, 0, 0

    pbar = tqdm(train_loader, ncols=80, desc='Training')

    for iteration, (img, true_age, level) in enumerate(pbar):
        img = img.to(device)
        true_age = true_age.to(device, dtype=torch.long)
        level = level.to(device)  # tensor shape: (batch, num_ages)

        # Perform a forward pass
        logits, probas = model(img)  # 'logits', 'probas' tensor shape: (batch, num_ages)

        # Calculate the training loss
        loss = loss_fn(logits, level, imp)
        train_loss += loss.item()

        # Calculate the current performance
        pred_level = probas > 0.5
        pred_age = torch.sum(pred_level, dim=1)
        num_examples += true_age.size(0)
        mae += torch.sum(torch.abs(pred_age - true_age))
        mse += torch.sum((pred_age - true_age) ** 2)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Updates the trainable weights
        optimizer.step()

    mae = mae.float() / num_examples
    mse = mse.float() / num_examples

    return train_loss / len(train_loader), mae, mse


# Validate the model over a single epoch
def validate_model(model, device, valid_loader, imp):
    model.eval()
    mae, mse, num_examples, valid_loss = 0, 0, 0, 0

    with torch.no_grad():
        for iteration, (img, true_age, level) in enumerate(valid_loader):
            img = img.to(device)
            true_age = true_age.to(device, dtype=torch.long)
            level = level.to(device)

            # Perform a forward pass
            logits, probas = model(img)

            # Calculate the training loss
            loss = loss_fn(logits, level, imp)
            valid_loss += loss.item()

            # Calculate the current performance
            pred_level = probas > 0.5
            pred_age = torch.sum(pred_level, dim=1)
            num_examples += true_age.size(0)
            mae += torch.sum(torch.abs(pred_age - true_age))
            mse += torch.sum((pred_age - true_age) ** 2)

    mae = mae.float() / num_examples
    mse = mse.float() / num_examples

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
    backbone = cfg['model_params']['backbone']
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
    print('Back-bone: ', backbone)

    # 4. Create a model
    model = RegressionModel(num_classes, backbone=backbone).to(device)
    if cfg['train_params']['resumed']:
        model.load_state_dict(torch.load(cfg['train_params']['weight_path'], device))
        print('Loading the saved checkpoint done. Training process resumed...')

    ages = get_ages(data_list_file=os.path.join(dataset_dir, train_list))
    imp = task_importance_weights(ages).to(device)  # tensor shape: (batch, num_ages)

    # 5. Specify loss function and optimizer
    optimizer = Adam(model.parameters(), lr=lr_start)

    # 6. Start training the model
    best_mae = 999
    for epoch in range(num_epochs):
        # 6.1. Train the model over a single epoch
        train_loss, train_mae, train_mse = train_model(model, device, optimizer, train_loader, imp)

        # 6.2. Validate the model
        valid_loss, valid_mae, valid_mse = validate_model(model, device, valid_loader, imp)

        print('Epoch: {} \tTraining MAE: {:.4f} \tValid MAE: {:.4f} \tTraining MSE: {:.4f} \tValid MSE: {:.4f}'.format(epoch, train_mae, valid_mae, train_mse, valid_mse))

        # 6.3. Save the model if the validation performance is increasing
        if best_mae >= valid_mae:
            print('Valid MAE decreased ({:.4f} --> {:.4f}). Model saved'.format(best_mae, valid_mae))
            torch.save(model.state_dict(),
                       './checkpoints/' + backbone + 'epoch_' + str(epoch) + '_mae_{0:.4f}'.format(valid_mae) + '.pt')
            best_mae = valid_mae
