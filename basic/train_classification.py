from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from model.model import RegressionModel, ClassificationModel, get_model
from torch.optim import Adam
import yaml
import torch.nn.functional as F
# Turn off the warning of YAML loader
import warnings

warnings.filterwarnings('ignore')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


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

    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    pbar = tqdm(train_loader, ncols=80, desc='Training')

    for iteration, (img, true_age, _) in enumerate(pbar):
        img = img.to(device)
        true_age = true_age.to(device, dtype=torch.long)

        # Perform a forward pass
        outputs = model(img)  # tensor shape: (batch, num_ages)

        # Calculate the training loss
        loss = loss_fn(outputs, true_age)
        cur_loss = loss.item()

        # Calculate the current performance
        _, pred_age = outputs.max(1)  # tensor shape: (batch)
        correct_num = pred_age.eq(true_age).sum().item()

        sample_num = img.size(0)
        loss_monitor.update(cur_loss, sample_num)
        accuracy_monitor.update(correct_num, sample_num)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Updates the trainable weights
        optimizer.step()

    return loss_monitor.avg, accuracy_monitor.avg


# Validate the model over a single epoch
def validate_model(model, device, loss_fn, valid_loader):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    for iteration, (img, true_age, _) in enumerate(valid_loader):
        img = img.to(device)
        true_age = true_age.to(device, dtype=torch.long)

        # Perform a forward pass
        outputs = model(img)
        preds.append(F.softmax(outputs, dim=-1).cpu().detach().numpy())
        gt.append(true_age.cpu().detach().numpy())

        # Calculate the valid loss
        loss = loss_fn(outputs, true_age)
        cur_loss = loss.item()

        # Calculate the current performance
        _, pred_age = outputs.max(1)
        correct_num = pred_age.eq(true_age).sum().item()

        sample_num = img.size(0)
        loss_monitor.update(cur_loss, sample_num)
        accuracy_monitor.update(correct_num, sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, num_classes)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


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
    tol_age = cfg['model_params']['tol_age']
    num_classes = int(max_age / age_group) + 1

    # 3. Load the dataset
    from utils.load_dataset import FaceDataset

    train_dataset = FaceDataset(dataset_dir=dataset_dir,
                                data_list_file=train_list,
                                input_shape=(img_size, img_size))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    valid_dataset = FaceDataset(dataset_dir=dataset_dir,
                                data_list_file=valid_list,
                                input_shape=(img_size, img_size))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    print('Number of training images: {}. Number of validation images: {}'.format(train_dataset.__len__(),
                                                                                  valid_dataset.__len__()))
    print('Age group: {}. Number of classes: {}'.format(age_group, num_classes))
    print('Back-bone: ', backbone)

    # 4. Specify the model and loss function
    # model = ClassificationModel(backbone=backbone, num_classes=num_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = get_model(num_classes=num_classes).to(device)

    # 5. Specify the optimizer
    optimizer = Adam(model.parameters(), lr=lr_start)

    # 6. Start training the model
    best_mae = 999
    for epoch in range(num_epochs):
        # 6.1. Train the model over a single epoch
        train_loss, train_acc = train_model(model, device, optimizer, loss_fn, train_loader)

        # 6.2. Validate the model
        val_loss, val_acc, val_mae = validate_model(model, device, loss_fn, valid_loader)

        print('Epoch: {} \tTraining accuracy: {:.4f} \tValid accuracy: {:.4f} \tValid MAE: {:.4f}'.format(epoch,
                                                                                                          train_acc,
                                                                                                          val_acc,
                                                                                                          val_mae))

        # 6.3. Save the model if the validation performance is increasing
        if val_mae < best_mae:
            print('Valid MAE decreased ({:.4f} --> {:.4f}). Model saved'.format(best_mae, val_mae))
            torch.save(model.state_dict(),
                       './checkpoints/epoch_' + str(epoch) + '_mae_{0:.4f}'.format(val_mae) + '.pt')
            best_mae = val_mae
