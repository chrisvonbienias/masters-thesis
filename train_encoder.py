from tqdm import tqdm
from time import time, localtime

import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import PointAutoEncoder
from dataset import ShapeNet
from logger import Logger
from utils.model_utils import  calc_cd

# hyperparameters
DEBUG = False
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_DENSE = 2048
NUM_WORKERS = 8
LOAD = False


# disable debug API for final training
# use cuDNN Autotuner
if (not DEBUG):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def train() -> None:
    
    data_path = "./resources/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # time stamp
    time_stamp = localtime(time())
    save_path = f'checkpoint/PointAutoEncoder/PointAutoEncoder_{time_stamp.tm_mon}_{time_stamp.tm_mday}_{time_stamp.tm_hour}:{time_stamp.tm_min}'

    # datasets
    train_dataset = ShapeNet(data_path, "train", num_dense=NUM_DENSE)
    validate_dataset = ShapeNet(data_path, "validate", num_dense=NUM_DENSE)
    print('Datasets loaded!')

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print('Dataloaders ready!')

    # model
    model = PointAutoEncoder(num_dense=NUM_DENSE).to(device)
    
    # load model.state_dict
    if LOAD:
        model.load_state_dict(torch.load('checkpoint/PointAutoEncoder/PointAutoEncoder_7_25_18:19.pth'))
    
    # optimizer
    # optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = Optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_schedual = Optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
    
    # logger
    logger = Logger(EPOCHS, model_name='PointAutoEncoder')
    print('Logger ready!')

    # training
    best_val = 0.13559938967227936
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epoch', position=0):
        
        train_loss = 0.0
        i = 0
        model.train()
        for c, p in tqdm(train_loader, desc='Training', position=1, leave=False):
            c = c.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # forward propagation
            features, dense_pred = model(c)
            dense_pred = dense_pred.to(device)

            # loss function
            loss = torch.mean(calc_cd(dense_pred, c)[0]).to(device)
            
            # back propagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            i += 1

        logger.train_data.append(float(train_loss / i))

        # evaluation
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            i = 0
            for c, p in tqdm(validate_loader, desc='Validation', position=1, leave=False):
                c = c.to(device)
                features, dense_pred = model(c)
                dense_pred = dense_pred.to(device)
                total_loss += torch.mean(calc_cd(dense_pred, c)[0])
                i += 1
        
        total_loss /= i
        logger.val_data.append(float(total_loss))
        lr_schedual.step(float(total_loss))
        
        if ((total_loss / i) < best_val):
            best_val = total_loss / i
            torch.save(model.state_dict(), f'{save_path}.pth')
            torch.save(torch.mean(features, dim=0), 'checkpoint/PointAutoEncoder/mean_features.pt')
            logger.best_data.append(1)
        else:
            logger.best_data.append(0)
            
        logger.save(save_path)
            
        # show values
        # print(f'Train loss: {logger.train_data[-1]}')
        # print(f'Val loss: {logger.val_data[-1]}')

    
if __name__ == "__main__":
    train()