import cv2
import numpy as np


def encodeImage(data):
    data = cv2.resize(data, (185, 268))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return bytes(data)


def decodeImage(data):
    decoded = np.frombuffer(data, dtype=np.uint8)
    decoded = decoded.reshape((268, 185, 3))
    print(decoded.shape)

    data = cv2.resize(decoded, (185, 268))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data


image = cv2.imread("./images/out.jpg")
cv2.imshow("original_image", image)
print(image.shape)

cv2.waitKey(0)

img_code = encodeImage(image)
img = decodeImage(img_code)
cv2.imshow("image_deirvlon", img)

cv2.waitKey(0)
