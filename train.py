from tqdm import tqdm
import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import PCN
from dataset import ShapeNet
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, emd_loss

# hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOSS = 'CD'

def train() -> None:
    
    data_path = "./resources/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # datasets
    train_dataset = ShapeNet(data_path, "train")
    validate_dataset = ShapeNet(data_path, "validate")
    print('Datasets loaded!')

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Dataloaders ready!')

    # model
    model = PCN(num_dense=16384, latent_dim=1024, grid_size=4).to(device)

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    # training
    train_step, val_step = 0, 0
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Learning...", leave=False):

        # hyperparameter alpha
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif epoch < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        print('Training model...')
        model.train()
        for i, (p, c) in enumerate(train_loader):
            p, c = p.to(device), c.to(device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p)

            # loss function
            if LOSS == 'CD':
                loss1 = cd_loss_L1(coarse_pred, c)
            elif LOSS == 'EMD':
                coarse_c = c[:, :1024, :]
                loss1 = emd_loss(coarse_pred, coarse_c)
            else:
                raise ValueError(f'Not implemented loss function {LOSS}')

            loss2 = cd_loss_L1(dense_pred, c)
            loss = loss1 + alpha * loss2
            print(f'Current loss: {loss}')

            # back propagation
            loss.backward()
            optimizer.step()

            train_step += 1

        lr_schedual.step()

        # evaluation
        print('Evaluating model...')
        model.eval()
        total_cd_l1 = 0.0

        with torch.no_grad():

            for i, (p, c) in enumerate(validate_loader):
                p, c = p.to(device), c.to(device)
                coarse_pred, dense_pred = model(p)
                total_cd_l1 += l1_cd(dense_pred, c).item()

        total_cd_l1 /= len(validate_dataset)
        val_step += 1
        print(f'Validation loss: {total_cd_l1}')

if __name__ == "__main__":
    train()