import numpy as np
from PIL import Image

# 1. Load emojis
emojis = [Image.open(f'set3/emoji{i}.png') for i in range(1, 9)]  # Load 8 images

# 2. Resize to 20x20
emojis = [emoji.resize((20, 20)) for emoji in emojis]

# 3. Convert to black and white (L mode)
emojis_bw = [emoji.convert('L') for emoji in emojis]

# 4. Concatenate images
widths, heights = zip(*(i.size for i in emojis_bw))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('L', (total_width, max_height))

x_offset = 0
for im in emojis_bw:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

# 5. Save emoji combination
new_im.save(f'concatenated_emojis3.png')

### In emojis.py change lines 21 to match image needed as well as logo_names