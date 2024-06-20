import json
import numpy as np
from MultiLayerPerceptron import train, Dense
from activation_functions import Sigmoid
from MultiLayerPerceptron import mse, mse_derivative

def fonts_to_bitmap(fonts:dict):
    bitmaps = {}
    for (character, hexaList) in fonts.items():
        bitmap = []
        for byte in hexaList:
            binary = format(byte, '08b')  
            row = [int(bit) for bit in binary[-5:]]  # Los caracteres tienen 3 bits de padding
            bitmap.extend(row)
        bitmaps[character] = bitmap
    return bitmaps


# Imprime un bitmap de 7x5
def print_bitmap(bitmap:list):
    for i in range(7):
        for j in range(5):
            print(bitmap[i*5 + j], end='')
        print()


# Devuelve una matriz de 7x5
def bitmap_as_matrix(bitmap:list): 
    return [[bitmap[i*5 + j] for j in range(5)] for i in range(7)]


def add_salt_and_pepper_noise_to_dataset(dataset, noiseLevel=0.3):
    noisy_dataset = dataset.copy()

    for i in range(len(noisy_dataset)):
        for j in range(len(noisy_dataset[i])):
            # Agregar ruido "salt"
            if np.random.rand() < noiseLevel:
                noisy_dataset[i, j] = 1

            # Agregar ruido "pepper"
            elif np.random.rand() < noiseLevel:
                noisy_dataset[i, j] = 0
        
    return noisy_dataset


def add_noise_to_dataset(dataset, noise_level=0.3):
    noisy_dataset = dataset.astype(np.float64)

    for i in range(len(noisy_dataset)):
        for j in range(len(noisy_dataset[i])):
             if np.random.rand() < noise_level:
                delta = np.random.normal(0, 0.5)
                if noisy_dataset[i, j] == 1.:
                    noisy_dataset[i, j] -= np.abs(delta)
                else:
                    noisy_dataset[i, j] += np.abs(delta)

    return noisy_dataset


def get_config_params(config_file: str):
    with open(config_file, 'r') as f:
        config = json.load(f)

    learning_rate = config["learning_rate"]

    max_epochs = config["max_epochs"]

    bias = config["bias"]

    beta1 = config["beta1"]

    beta2 = config["beta2"]

    epsilon = config["epsilon"]

    optimizer = config["optimizer"]

    activation = config["activation"]

    return learning_rate, max_epochs, bias, beta1, beta2, epsilon, optimizer, activation


def train_different_architectures(optimizer, learning_rate, max_epochs, dataset):
    mse_list = []

    # 35-20-10-2-10-20-35
    autoencoder = [
    Dense(35, 20, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(20, 10, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(10, 2, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(2, 10, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(10, 20, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(20, 35, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    ]   
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    # 35-15-5-2-5-15-35
    autoencoder = [
    Dense(35, 15, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(15, 5, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(5, 2, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(2, 5, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(5, 15, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(15, 35, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    ]   
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    # 35-15-2-15-35
    autoencoder = [
    Dense(35, 15, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(15, 2, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(2, 15, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(15, 35, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    ]
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    # 35-10-2-10-35
    autoencoder = [
    Dense(35, 10, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(10, 2, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(2, 10, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    Dense(10, 35, optimizer_type=optimizer, learning_rate=learning_rate),
    Sigmoid(),
    ]
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    return mse_list

def compare_matrixes(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Las matrices deben tener las mismas dimensiones")

    count = 0

    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] == matrix2[i][j]:
                count += 1

    return count