from scipy import ndimage, datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# image = datasets.image().astype('int32')

image = Image.open('./assets/monkey.jpeg')
image = np.asarray(image).astype(np.int32)

# print(image)
sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
sobel_v = ndimage.sobel(image, 1)  # vertical gradient
magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
magnitude *= 255.0 / np.max(magnitude)  # normalization
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.gray()  # show the filtered result in grayscale
axs[0, 0].imshow(image)
axs[0, 1].imshow(sobel_h)
axs[1, 0].imshow(sobel_v)
axs[1, 1].imshow(magnitude)
titles = ["original", "horizontal", "vertical", "magnitude"]
for i, ax in enumerate(axs.ravel()):
    ax.set_title(titles[i])
    ax.axis("off")
plt.savefig('sobel.png')
plt.show()