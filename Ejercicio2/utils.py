import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(path):
    # Abre la imagen y conviértela a escala de grises
    image = Image.open(path).convert('L')
    # Convierte la imagen a una matriz NumPy
    matrix = np.array(image)
    return matrix


def save_image(matrix, path):
    # Crea una imagen a partir de la matriz
    image = Image.fromarray(matrix.astype(np.uint8), mode='L')
    # Guarda la imagen en la ruta especificada como formato PNG
    image.save(path, format='PNG')


def get_all_images(path):
    images = []

    for archivo in os.listdir(path):
        # Verifica si el archivo es un archivo PNG
        if archivo.endswith(".png"):
            # Construye la ruta completa del archivo
            full_path = os.path.join(path, archivo)

            # Carga la imagen en una matriz y agrégala a la lista
            matrix = load_image(full_path)
            images.append(matrix)
    return images


def plot_data(original, decoded, input_rows, input_cols):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decoded')
    ax1.imshow(np.array(original).reshape((input_rows, input_cols)), cmap='gray')
    ax2.imshow(np.array(decoded).reshape((input_rows, input_cols)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.show()