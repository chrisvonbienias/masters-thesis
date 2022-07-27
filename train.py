from tqdm import tqdm
from time import time, localtime

import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import PCN, FinerPCN, PointAutoEncoder, MSN
from dataset import ShapeNet
from logger import Logger
from utils.model_utils import calc_dcd, calc_cd

# hyperparameters
DEBUG = False
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOSS = 'CD'
MODEL_NAME = 'FinerPCN'
NUM_DENSE = 8192
NUM_WORKERS = 8
ALPHA = 0.5
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
    save_path = f'checkpoint/{MODEL_NAME}/{MODEL_NAME}_{LOSS}_{time_stamp.tm_mon}_{time_stamp.tm_mday}_{time_stamp.tm_hour}:{time_stamp.tm_min}'

    # datasets
    train_dataset = ShapeNet(data_path, "train", num_dense=NUM_DENSE)
    validate_dataset = ShapeNet(data_path, "validate", num_dense=NUM_DENSE)
    print('Datasets loaded!')

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print('Dataloaders ready!')

    # model
    if MODEL_NAME == 'PCN':
        model = PCN(num_dense=NUM_DENSE, latent_dim=1024, grid_size=4).to(device)
    elif MODEL_NAME == 'FinerPCN':
        model = FinerPCN(num_dense=NUM_DENSE, grid_size=4, grid_scale=0.5).to(device)
    elif MODEL_NAME == 'PointAutoEncoder':
        model = PointAutoEncoder(num_dense=NUM_DENSE).to(device)
    elif MODEL_NAME == 'MSN':
        model = MSN(num_points=NUM_DENSE).to(device)
    else:
        raise ValueError(f'Model {MODEL_NAME} not implemented')
    
    # load model.state_dict
    if LOAD:
        model.load_state_dict(torch.load('checkpoint/PCN/PCN_DCD_7_26_8:9.pth'))
    
    # optimizer
    # optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = Optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_schedual = Optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
    
    # logger
    logger = Logger(EPOCHS, model_name=MODEL_NAME)
    print('Logger ready!')

    # training
    best_val = 100
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epoch', position=0):
        
        train_loss = 0.0
        i = 0
        model.train()
        for c, p in tqdm(train_loader, desc='Training', position=1, leave=False):
            c, p = c.to(device), p.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # forward propagation
            if MODEL_NAME == 'MSN':
                coarse_pred, dense_pred, loss_exp = model(p)
                loss_exp = loss_exp.to(device)
            else:
                coarse_pred, dense_pred = model(p)
                
            coarse_pred = coarse_pred.to(device)
            dense_pred = dense_pred.to(device)

            # loss function
            if LOSS == 'CD':
                loss1 = torch.mean(calc_cd(coarse_pred, c)[0]).to(device)
                loss2 = torch.mean(calc_cd(dense_pred, c)[0]).to(device)
            elif LOSS == 'DCD':
                loss1 = torch.mean(calc_dcd(coarse_pred, c)[0]).to(device)
                loss2 = torch.mean(calc_dcd(dense_pred, c)[0]).to(device)
            else:
                raise ValueError(f'Not implemented loss function {LOSS}')

            if MODEL_NAME == 'MSN':
                loss = loss1 + 0.1 * loss_exp + loss2
            else:
                loss = loss1 + ALPHA * loss2
            
            # back propagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            i += 1

        logger.train_data.append(float(train_loss / i))

        # evaluation
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            i = 0
            for c, p in tqdm(validate_loader, desc='Validation', position=1, leave=False):
                c, p = c.to(device), p.to(device)
                coarse_pred, dense_pred = model(p)
                coarse_pred = coarse_pred.to(device)
                dense_pred = dense_pred.to(device)
                total_loss += torch.mean(calc_cd(dense_pred, c)[0]).item()
                i += 1
        
        total_loss /= i
        logger.val_data.append(float(total_loss))
        lr_schedual.step(float(total_loss))
        
        if ((total_loss / i) < best_val):
            best_val = total_loss / i
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