import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from models import PCN
from dataset import ShapeNet

# hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train() -> None:
    
    data_path = "./resources/dataset"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # datasets
    train_dataset = ShapeNet(data_path, "train")
    validate_dataset = ShapeNet(data_path, "validate")

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model
    model = PCN(num_dense=16384, latent_dim=1024, grid_size=4).to(device)

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training
    for epoch in range(1, EPOCHS + 1):
        


if __name__ == "__main__":
    train()