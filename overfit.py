import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import PCN, FinerPCN, PointAutoEncoder, MSN, CRN_Generator
from dataset import ShapeNet
from utils.model_utils import  calc_cd

# hyperparameters
EPOCHS = 10000
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_DENSE = 8192
NUM_WORKERS = 8
ALPHA = 0.5
LOAD = False

def overfit() -> None:
    
    data_path = "./resources/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # datasets
    train_dataset = ShapeNet(data_path, "validate", num_dense=NUM_DENSE)
    print('Dataset loaded!')

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    c, p = next(iter(train_loader))
    print('Dataloader ready!')

    # model
    model = MSN().to(device)
    
    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training
    model.train()
    for epoch in range(1, EPOCHS + 1):
        
        c, p = c.to(device), p.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # forward propagation
        coarse_pred, dense_pred, exp_loss = model(p)
            
        #coarse_pred = coarse_pred.to(device)
        dense_pred = dense_pred.to(device)
        coarse_pred = coarse_pred.to(device)
        exp_loss = exp_loss.to(device)

        # loss function
        loss1 = torch.mean(calc_cd(coarse_pred, c)[0]).to(device)
        loss2 = torch.mean(calc_cd(dense_pred, c)[0]).to(device)
        loss = loss1 + ALPHA * loss2 + 0.1 * exp_loss
        
        # back propagation
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch} | Loss: {loss.item()}', end='\r')

    
if __name__ == "__main__":
    overfit()