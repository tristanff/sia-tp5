import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Print two 7x5 matrices, one with the original character and the other with the predicted character
def plot_bitmap_matrix(original, predicted, character):
    # Create a heatmap using imshow from matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(3, 2))  # 1 row, 2 columns

    # Edge case
    if character == "DEL":
        colors1 = ['black']
        custom_cmap1 = plt.matplotlib.colors.ListedColormap(colors1)
        colors2 = ['gray', 'black']
        custom_cmap2 = plt.matplotlib.colors.ListedColormap(colors2)

        axs[0].imshow(original, cmap=custom_cmap1, interpolation='none')
        axs[1].imshow(predicted, cmap=custom_cmap2, interpolation='none')
    else:
        axs[0].imshow(original, cmap='binary', interpolation='none')
        axs[1].imshow(predicted, cmap='binary', interpolation='none')

    # Create heatmaps for each pair of matrices
    axs[0].set_title('Original ' + character)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].set_title('Predicted ' + character)
    axs[1].set_xticks([])
    axs[1].set_yticks([])


# Print a list of 7x5 matrices, one for each character
def plot_bitmap_matrix_2(matrix_list, character_list, title):
    num_plots = len(matrix_list)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(np.ceil(num_plots / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_plots:
                axes[i, j].imshow(matrix_list[index], cmap='binary', interpolation='none', vmin=0, vmax=1)
                axes[i, j].set_title(character_list[index])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                fig.delaxes(axes[i, j])  # Remove axes if no more plots

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_latent_spaces(latent_space, characters):
    # Convert the list of tuples and labels into a Pandas DataFrame
    df = pd.DataFrame({'x': [p[0] for p in latent_space], 'y': [p[1] for p in latent_space], 'label': characters})

    # Plot the points using Pandas
    ax = df.plot.scatter(x='x', y='y', color='blue', marker='o', s=50)

    # Annotate each point with its corresponding label
    for i, row in df.iterrows():
        ax.annotate(row['label'], (row['x'], row['y']), textcoords="offset points", xytext=(0,5), ha='center')

    # Configure labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Latent Space Plot for each Character')

    # Show the plot
    plt.grid()
    plt.show()
