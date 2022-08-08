from tqdm import tqdm
from time import time, localtime

import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import CRN_Generator, CRN_Discriminator
from dataset import ShapeNet
from logger import Logger
from utils.model_utils import  calc_cd, calc_dcd

# hyperparameters
DEBUG = False
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.001
LOSS = 'DCD'
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
else:
    torch.autograd.set_detect_anomaly(True)
    torch.autograd.profiler.profile(True)
    torch.autograd.profiler.emit_nvtx(True)
    
    
def train() -> None:
    
    data_path = "./resources/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    
    # time stamp
    time_stamp = localtime(time())
    save_path_gen = f'checkpoint/CRN/CRN_GEN_{LOSS}_{time_stamp.tm_mon}_{time_stamp.tm_mday}_{time_stamp.tm_hour}:{time_stamp.tm_min}'
    save_path_disc = f'checkpoint/CRN/CRN_DICS_{LOSS}_{time_stamp.tm_mon}_{time_stamp.tm_mday}_{time_stamp.tm_hour}:{time_stamp.tm_min}'

    # datasets
    train_dataset = ShapeNet(data_path, "train", num_dense=NUM_DENSE)
    validate_dataset = ShapeNet(data_path, "validate", num_dense=NUM_DENSE)
    print('Datasets loaded!')

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print('Dataloaders ready!')

    # model
    model_gen = CRN_Generator(num_dense=NUM_DENSE).to(device)
    model_disc = CRN_Discriminator().to(device)
    
    # load model.state_dict
    if LOAD:
        model_gen.load_state_dict(torch.load('checkpoint/CRN/CRN_GEN_DCD_8_1_20:0.pth'))
        model_disc.load_state_dict(torch.load('checkpoint/CRN/CRN_DICS_DCD_8_1_20:0.pth'))
    
    # optimizer
    # optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_gen = Optim.AdamW(model_gen.parameters(), lr=LEARNING_RATE)
    optimizer_disc = Optim.AdamW(model_disc.parameters(), lr=LEARNING_RATE)
    # lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_schedual_gen = Optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen, 'min', patience = 5)
    lr_schedual_disc = Optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, 'min', patience = 5)
    
    # logger
    logger_gen = Logger(EPOCHS, model_name='CRN_Generator')
    logger_disc = Logger(EPOCHS, model_name='CRN_Discriminator')
    print('Logger ready!')

    # training
    best_val_gen = 100 
    best_val_disc = 100
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epoch', position=0):
        
        # training
        train_loss_gen = 0.0
        train_loss_disc = 0.0
        i = 0
        model_gen.train()
        model_disc.train()
        for c, p in tqdm(train_loader, desc='Training', position=1, leave=False):
            c, p = c.to(device), p.to(device)
            
            ### GENERATOR ###
                        
            # forward propagation
            coarse_pred, dense_pred = model_gen(p)
            coarse_pred = coarse_pred.to(device)
            dense_pred = dense_pred.to(device)

            # loss function
            loss1 = torch.mean(calc_dcd(dense_pred, c)[0]).to(device)
            loss2 = torch.mean(calc_dcd(coarse_pred, c)[0]).to(device)
            loss_gen = loss1 + ALPHA * loss2

            
            ### DISCRIMINATOR
            
            # forward propagation
            d_fake = model_disc(dense_pred)
            d_fake = d_fake.to(device)
            d_real = model_disc(c.detach())
            d_real = d_real.to(device)
            
            # loss function
            loss_fake = torch.mean(d_fake ** 2)
            loss_real = torch.mean((d_real - 1) ** 2)
            loss_disc1 = 0.5 * (loss_fake + loss_real)
            loss_disc2 = torch.mean((loss_fake - 1) ** 2)
            
            ### TOTAL LOSS
            loss_gen  = loss_disc2 + loss_gen
            loss_disc = loss_disc1
            
            ### BACK PROPAGATION
            optimizer_gen.zero_grad(set_to_none=True)
            loss_gen.backward(retain_graph=True)
            
            optimizer_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            
            optimizer_gen.step()
            optimizer_disc.step()
            
            train_loss_gen += loss_gen.item()
            train_loss_disc += loss_disc.item()
            
            i += 1

        logger_gen.train_data.append(train_loss_gen / i)
        logger_disc.train_data.append(train_loss_disc / i)

        # evaluation
        model_gen.eval()
        model_disc.eval()
        total_loss_gen = 0.0
        total_loss_disc = 0.0

        with torch.no_grad():
            i = 0
            for c, p in tqdm(validate_loader, desc='Validation', position=1, leave=False):
                c, p = c.to(device), p.to(device)
                
                # generator
                coarse_pred, dense_pred = model_gen(p)
                dense_pred = dense_pred.to(device)
                
                # discriminator
                d_fake = model_disc(dense_pred)
                d_fake = d_fake.to(device)
            
                total_loss_gen += torch.mean(calc_cd(dense_pred, c)[0]).item()
                total_loss_disc += torch.mean((loss_fake - 1) ** 2).item()
                i += 1
        
        total_loss_gen /= i
        total_loss_disc /= i
        logger_gen.val_data.append(total_loss_gen)
        logger_disc.val_data.append(total_loss_disc)
        #lr_schedual_gen.step(total_loss_gen)
        #lr_schedual_disc.step(total_loss_disc)
        
        if ((total_loss_gen / i) < best_val_gen):
            best_val_gen = total_loss_gen / i
            torch.save(model_gen.state_dict(), f'{save_path_gen}.pth')
            logger_gen.best_data.append(1)
        else:
            logger_gen.best_data.append(0)
            
        if ((total_loss_disc / i) < best_val_disc):
            best_val_disc = total_loss_disc / i
            torch.save(model_disc.state_dict(), f'{save_path_disc}.pth')
            logger_disc.best_data.append(1)
        else:
            logger_disc.best_data.append(0)
            
        logger_gen.save(save_path_gen)
        logger_disc.save(save_path_disc)
            
        # show values
        # print(f'Train loss: {logger.train_data[-1]}')
        # print(f'Val loss: {logger.val_data[-1]}')

    
if __name__ == "__main__":
    train()