import tensorflow as tf
import numpy as np
import cv2
import os
import imutils

# path to train folder
train_path = "./dataset/train"
# see how many total labels are there
labels = os.listdir(train_path)
# if no labels are found then quit the program
if not labels:
    print("No labels found !")
    quit()
# read the image, here you need to specify the full path to the image
#input if format is image
img = cv2.imread(r"D:\ROBOTICS\viverr_1\samples\download.jfif")
original = img.copy()
original = imutils.resize(original, width=720)
if img is None:
    print("bad image")
    quit()
# here we need to pre-process the image to predict
img = cv2.resize(img, (224, 224))
img = img / 255
img = img.astype("float32")
img = np.expand_dims(img, axis=0)
# load the trained model from saved_model folder
model = tf.keras.models.load_model("./saved_models/1586596294.9131072.h5")
# we get the prediction here
p = model.predict(img)[0]
# we see the softmax out put of the activated label
ind = np.argmax(p)
# calculate accuracy and convert it to percentage
acc = int(p[ind] * 100)
# map its value
value = labels[ind]
# display
if value == "rose":
    cv2.putText(original, value + " " + str(acc) + "%", (50, 50), 2, cv2.FONT_HERSHEY_DUPLEX, (0, 255, 0), 2)
else:
    cv2.putText(original, "Not Rose" + " " + str(acc) + "%", (50, 50), 2, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 2)
cv2.imshow("preview", original)
cv2.waitKey(0)
