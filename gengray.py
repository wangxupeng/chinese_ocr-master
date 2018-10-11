from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    img = Image.open('/home/xiaoke/terhome/test.jpeg')
    img = np.array(img)
    if img.ndim == 3:
        img = img[:, :, 0]

    plt.imshow(img, cmap=plt.cm.gray_r)
    plt.show()
