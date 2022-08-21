import matplotlib.pyplot as plt
import pandas as pd

class LoggerExtra():
    
    def __init__(self, epochs, disc=False):
        
        self.epochs = range(1, epochs)
        self.model_name = 'CPN'
        self.disc = disc
        self.train_data = []
        self.val_data = []
        self.best_data = []
        self.coarse_loss = []
        self.dense_loss = []
        self.exp_loss = []
        self.disc_loss = []
    
    def save(self, save_path):
        
        if self.disc:
            columns = {
                "Coarse loss": self.coarse_loss,
                "Dense loss": self.dense_loss,
                "Expansion loss": self.exp_loss,
                "Discriminator loss": self.disc_loss,
                "Training loss": self.train_data,
                "Validation loss": self.val_data,
                "Best": self.best_data
            }
        else:
            columns = {
                "Coarse loss": self.coarse_loss,
                "Dense loss": self.dense_loss,
                "Expansion loss": self.exp_loss,
                "Training loss": self.train_data,
                "Validation loss": self.val_data,
                "Best": self.best_data
            }
            
        data = pd.DataFrame(columns)
        path = f'{save_path}.csv'
        data.to_csv(path)
        