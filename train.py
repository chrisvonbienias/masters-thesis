from cmath import inf
from tqdm import tqdm
import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader
from time import time, localtime

from models import PCN
from dataset import ShapeNet
from logger import Logger
from utils.model_utils import calc_dcd, calc_emd, calc_cd

# hyperparameters
DEBUG = True
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOSS = 'CD'
MODEL_NAME = 'PCN'
NUM_DENSE = 8192


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
    save_path = f'checkpoint/{MODEL_NAME}_{time_stamp.tm_mon}_{time_stamp.tm_mday}_{time_stamp.tm_hour}:{time_stamp.tm_min}'

    # datasets
    train_dataset = ShapeNet(data_path, "train", num_dense=NUM_DENSE)
    validate_dataset = ShapeNet(data_path, "validate", num_dense=NUM_DENSE)
    print('Datasets loaded!')

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print('Dataloaders ready!')

    # model
    if MODEL_NAME == 'PCN':
        model = PCN(num_dense=NUM_DENSE, latent_dim=1024, grid_size=4).to(device)
    else:
        raise ValueError(f'Model {MODEL_NAME} not implemented')
    
    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    
    # logger
    logger = Logger(EPOCHS, model_name=MODEL_NAME)
    print('Logger ready!')

    # training
    best_val = 100
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epoch', position=0, leave=False):

        train_loss = 0.0
        i = 0
        model.train()
        for c, p in tqdm(train_loader, desc='Training', position=1):
            c, p = c.to(device), p.to(device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p)
            coarse_pred = coarse_pred.cuda()
            coarse_pred = dense_pred.cuda()

            # loss function
            if LOSS == 'CD':
                loss = torch.mean(calc_cd(dense_pred, c, calc_f1=True)[2]).cuda()
            elif LOSS == 'EMD':
                loss = torch.mean(calc_emd(dense_pred, c)).cuda()
            elif LOSS == 'DCD':
                loss = torch.mean(calc_dcd(dense_pred, c)[0]).cuda()
            else:
                raise ValueError(f'Not implemented loss function {LOSS}')

            # back propagation
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            i += 1

        logger.train_data.append(float(train_loss / i))
        lr_schedual.step()

        # evaluation
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            i = 0
            for c, p in tqdm(validate_loader, desc='Validation', position=1):
                c, p = c.to(device), p.to(device)
                coarse_pred, dense_pred = model(p)
                coarse_pred = coarse_pred.cuda()
                coarse_pred = dense_pred.cuda()
                total_loss += torch.mean(calc_cd(dense_pred, c, calc_f1=True)[2])
                i += 1
        
        logger.val_data.append(float(total_loss / i))
        
        if (total_loss / i < best_val):
            best_val = total_loss / i
            # print(f'Saving model...')
            torch.save(model.state_dict(), f'{save_path}.pth')
            logger.save(save_path)
        
        # show values
        # print(f'Train loss: {logger.train_data[-1]}')
        # print(f'Val loss: {logger.val_data[-1]}')

    
if __name__ == "__main__":
    train()