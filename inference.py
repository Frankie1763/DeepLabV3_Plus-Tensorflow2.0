import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from deeplab import DeepLabV3Plus
import tensorflow as tf
import cv2
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input
import argparse

print('Tensorflow', tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/\
                            deeplab_v3_plus_tensorflow_v2/checkpoints/training_new_1/cp-0148.ckpt",
                    help="trained model path")
parser.add_argument("--output", type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/\
                            deeplab_v3_plus_tensorflow_v2/val_output_new",
                    help="output path")
parser.add_argument("--img_txt", type=str,
                    default="/content/drive/My Drive/CS Internship/\
                    DeepLab_v3/deeplab_v3_tensorflow_v1/dataset/test_img_full_path.txt",
                    help="text file that contains full path of test images")
parser.add_argument("--msk_txt", type=str,
                    default="/content/drive/My Drive/CS Internship/\
                    DeepLab_v3/deeplab_v3_tensorflow_v1/dataset/test_msk_full_path.txt",
                    help="text file that contains full path of test masks")

# global variables
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor, 21=boundary
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (255,255,255)]

batch_size = 10
H, W = 512, 512
num_classes = 22  # including background and the boundary pixels
_DEPTH = 3

def create_list(img_txt,msk_txt):
    with open(img_txt,"r") as f:
        img_lst = [line[:-1] for line in f.readlines()]
    with open(msk_txt,"r") as f :
        msk_lst = [line[:-1] for line in f.readlines()]
    return img_lst, msk_lst

def load_model(model_path):
    model = DeepLabV3Plus(H, W, num_classes)
    model.load_weights(model_path)
    return model

def pipeline(image, model, save_img=False, save_dir=None, filename=None):
    global b
    alpha = 0.5
    dims = image.shape
    image = cv2.resize(image, (W, H))
    x = image.copy()
    z = model.predict(preprocess_input(np.expand_dims(x, axis=0)))
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)

    img_color = image.copy()
    for i in np.unique(y):
        if i <= 21:  # exclude the boundary pixels
            img_color[y == i] = label_colours[i]
    disp = img_color.copy()
    cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)

    if save_img:
        output = img_color
        cv2.imwrite(save_dir + filename, output)
    else:
        plt.figure(figsize=(20, 10))
        # out = np.concatenate([image/255, img_color/255, disp/255], axis=1)

        plt.imshow(img_color / 255.0)
        # plt.imshow(out)

def predict_label(model, img_path): 
  img = img_to_array(load_img(img_path))
  img = cv2.resize(img, (W,H))
  img = tf.expand_dims(img, axis=0)
  img = preprocess_input(img)
  pred_label = model.predict(img)
  pred_label = np.squeeze(pred_label)
  pred_label = np.argmax(pred_label,axis=2)
  return pred_label

def draw_masks(img_lst, model, output):
    for img_path in tqdm(img_lst):
        img = img_to_array(load_img(img_path))
        pipeline(img, model, filename=img_path[-15:], save_dir=output, save_img=True)

def main():
    FLAGS, unparsed = parser.parse_known_args()
    model = load_model(FLAGS.model)
    img_lst, msk_lst = create_list(FLAGS.img_txt, FLAGS.msk_txt)
    draw_masks(img_lst, model, FLAGS.output)

