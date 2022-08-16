
from tqdm import tqdm
from time import time, localtime

import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import CPN
from dataset import ShapeNetLabel
from logger import LoggerExtra
from utils.model_utils import calc_dcd

# hyperparameters
DEBUG = False
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_DENSE = 4096
NUM_WORKERS = 4
ALPHA = 0.5
LOAD = False
SUBSET = 10

torch.autograd.set_detect_anomaly(True)

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
    save_path = f'checkpoint/CPN/CPN_{time_stamp.tm_mon}_{time_stamp.tm_mday}_{time_stamp.tm_hour}:{time_stamp.tm_min}'

    # datasets
    train_dataset = ShapeNetLabel(data_path, "train", num_dense=NUM_DENSE, subset=SUBSET)
    validate_dataset = ShapeNetLabel(data_path, "validate", num_dense=NUM_DENSE, subset=SUBSET)
    print('Datasets loaded!')

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print('Dataloaders ready!')

    # model
    model = CPN(num_dense=NUM_DENSE).to(device)
    
    # load model.state_dict
    if LOAD:
        model.load_state_dict(torch.load('checkpoint/MSN/MSN_DCD_8_3_14:21.pth'))
    
    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = Optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_schedual = Optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10, verbose=True)
    
    # logger
    logger = LoggerExtra(EPOCHS)
    print('Logger ready!')

    # training
    best_val = 100
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epoch', position=0):
        
        train_loss = 0.0
        coarse_loss = 0.0
        dense_loss = 0.0
        exp_losss = 0.0
        i = 0
        model.train()
        for c, p, label in tqdm(train_loader, desc='Training', position=1, leave=False):
            c, p, label = c.to(device), p.to(device), label.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # forward propagation
            coarse_pred, dense_pred, exp_loss = model((label, p))
                
            coarse_pred = coarse_pred.to(device)
            dense_pred = dense_pred.to(device)
            exp_loss = exp_loss.to(device)

            # loss function
            loss1 = torch.mean(calc_dcd(coarse_pred, c)[0]).to(device)
            loss2 = torch.mean(calc_dcd(dense_pred, c)[0]).to(device)
            loss = loss1 + ALPHA * loss2 + exp_loss
            
            # back propagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            coarse_loss += loss1.item()
            dense_loss += loss2.item()
            exp_losss += exp_loss.item()
            i += 1
            
        train_loss = train_loss / i
        coarse_loss = coarse_loss / i
        dense_loss = dense_loss / i
        exp_losss = exp_losss / i
        logger.train_data.append(train_loss)
        logger.coarse_loss.append(coarse_loss)
        logger.dense_loss.append(dense_loss)
        logger.exp_loss.append(exp_losss)

        # evaluation
        model.eval()
        total_loss = 0.0

        i = 0
        with torch.no_grad():
            
            for c, p, label in tqdm(validate_loader, desc='Validation', position=1, leave=False):
                c, p, label = c.to(device), p.to(device), label.to(device)
                coarse_pred, dense_pred, _ = model((label, p))
                dense_pred = dense_pred.to(device)
                total_loss += torch.mean(calc_dcd(dense_pred, c)[0]).item()
                i += 1
        
        total_loss = total_loss / i
        logger.val_data.append(total_loss)
        lr_schedual.step(total_loss)
        
        if (total_loss < best_val):
            best_val = total_loss
            torch.save(model.state_dict(), f'{save_path}.pth')
            logger.best_data.append(1)
        else:
            logger.best_data.append(0)
            
        logger.save(save_path)
            
        # show values
        # print(f'Train loss: {logger.train_data[-1]}')
        # print(f'Val loss: {logger.val_data[-1]}')

    
if __name__ == "__main__":
    train()