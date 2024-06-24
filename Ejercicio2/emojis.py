import numpy as np
from PIL import Image

logos_names = [
    "grinning",
    "smile",
    "laughing",
    "joy",
    "wink",
    "expressionless",
    "frown",
    "gasp",
    "disappointed",
    "angry",
]

emoji_size = (24, 24)
emoji_images = []

def load_logos_images():
    img = np.asarray(Image.open('emojis2.png').convert("L"))
    logoss_per_row = img.shape[1] / emoji_size[1]
    for i in range(len(logos_names)):
        y = int((i // logoss_per_row) * emoji_size[0])
        x = int((i % logoss_per_row) * emoji_size[1])
        logos_matrix = img[y:(y + emoji_size[1]), x:(x + emoji_size[0])] / 255
        logos_vector = logos_matrix.flatten()
        emoji_images.append(logos_vector)

load_logos_images()
