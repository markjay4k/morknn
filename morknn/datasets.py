import pandas as pd
import numpy as np


def mnist_dataset(split_factor=0.05, verbose=False):
    """
        reads the mnist dataset and returns training and validation data
        
        split_factor (float): (0 to 1) defines the size of the validation data
                              0 = no validation data, 1 = 100% validation data
                              default = 0.05
        return:
          - train_x (np.ndarray): array of flattened images
          - train_y (np.ndarray): array of labels
          -   val_x (np.ndarray): array of flattened images
          -   val_y (np.ndarray): array of labels
          
    """
    data_df = pd.read_csv(f'./datasets/mnist/train.csv')
    data_x = data_df[data_df.columns[1:]].to_numpy() / 255
    data_x = data_x.astype(np.float32)
    data_y = data_df['label'].to_numpy()
    data_y = data_y.astype(np.int32)
    split_index = int(data_y.shape[0] * split_factor)
    train_x = data_x[split_index:, :].transpose()
    val_x = data_x[:split_index, :].transpose()
    train_y = data_y[split_index:]
    val_y = data_y[:split_index]
    if verbose:
        print(f'train_x shape = {train_x.shape}')
        print(f'train_y shape = {train_y.shape}')
        print(f'  val_x shape = {val_x.shape}')
        print(f'  val_x shape = {val_y.shape}')
    return train_x, train_y, val_x, val_y