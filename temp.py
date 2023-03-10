import cv2
img = cv2.imread('dataset/kolektor2/images/train/10934.png')
print(img.shape)




img_1= cv2.copyMakeBorder(
    img[:,:,0],
    0,
    10,
    0,
    10,
    cv2.BORDER_CONSTANT,
    value=(255,255,255))

print(img_1.shape)