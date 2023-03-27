from PIL import Image
import gzip
import os

# set paths to MNIST files
image_file = 't10k-images-idx3-ubyte.gz'
label_file = 't10k-labels-idx1-ubyte.gz'
output_dir = 'mnist/testing'

with gzip.open(label_file, 'rb') as f:
    labels = f.read()[8:]
with gzip.open(image_file, 'rb') as f:
    data = f.read()
    offset = 16  # skip header
    width, height = 28, 28
    num_images = 10000
    for i in range(num_images):
        image = Image.frombytes('L', (width, height), data[offset:offset+width*height])
        label = labels[i]
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image.save(os.path.join(label_dir, f'{i}.png'))
        offset += width * height