import matplotlib.pyplot as plt
import pandas as pd
from time import localtime, time

class Logger():
    
    def __init__(self, epochs, model_name='PCN', test=False):
        
        self.epochs = range(1, epochs)
        self.model_name = model_name
        self.test = test
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.best_data = []
  
        
    def plot(self, epoch):
        
        if self.test:
            plt.plot(range(1, epoch), self.train_data, 'g', label='Training loss')
            plt.plot(self.epochs, self.val_data, 'b', label='Validation loss')
            plt.title('Training and Validation loss')
            plt.legend()
        else:
            plt.plot(range(1, epoch), self.test_data, 'g')
            plt.title('Training loss')
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
    
    def save(self, save_path):
        
        if self.test:
            columns = {
                "Test loss": self.test_data,
                "Best": self.best_data
            }
        else:
            columns = {
                "Training loss": self.train_data,
                "Validation loss": self.val_data,
                "Best": self.best_data
            }
            
        data = pd.DataFrame(columns)
        path = f'{save_path}.csv'
        data.to_csv(path)
        