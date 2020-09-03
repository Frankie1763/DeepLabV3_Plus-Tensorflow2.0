from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image,ImageDraw, ImageFont

import argparse

print('Tensorflow', tf.__version__)

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_plus_tensorflow_v2/checkpoints/training_new_1/cp-0148.ckpt",
                    help="trained model path")
parser.add_argument("--output", type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_plus_tensorflow_v2/val_output_new/",
                    help="output path")
parser.add_argument("--img_txt", type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_tensorflow_v1/dataset/val_img_full_path.txt",
                    help="text file that contains full path of test images")
parser.add_argument("--msk_txt", type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_tensorflow_v1/dataset/val_msk_full_path.txt",
                    help="text file that contains full path of test masks")
parser.add_argument('--backbone', type=str,
                    default="resnet50",
                    help='resnet50/resnet101/xception/resnet50_duc')

# global variables
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor, 21=boundary
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)]

batch_size = 10
H, W = 512, 512
num_classes = 22  # including background and the boundary pixels
_DEPTH = 3


def create_list(img_txt, msk_txt):
    with open(img_txt, "r") as f:
        img_lst = [line[:-1] for line in f.readlines()]
    with open(msk_txt, "r") as f:
        msk_lst = [line[:-1] for line in f.readlines()]
    return img_lst, msk_lst


def load_model(model_path, backbone):
    if backbone == "resnet50":
        from deeplab_resnet50 import DeepLabV3Plus
    elif backbone == "resnet101":
        from deeplab_resnet101 import DeepLabV3Plus
    elif backbone == "xception":
        from deeplab_xception import DeepLabV3Plus
    elif backbone == "renet50_duc":
        from deeplab_resnet50_duc import DeepLabV3Plus
    model = DeepLabV3Plus(H, W, num_classes)
    model.load_weights(model_path)
    print("successfully load model")
    return model


def pipeline(image, gt, model, save_img=False, save_dir=None, filename=None):
    # predict
    alpha = 0.5
    dims = image.shape
    image = cv2.resize(image, (W, H))
    gt = cv2.resize(gt, (W, H), interpolation = cv2.INTER_NEAREST)  # use the nearest pixel instead of bilinear
    x = image.copy()
    z = model.predict(preprocess_input(np.expand_dims(x, axis=0)))
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)
    p_a = pixel_accuracy(y, gt)

    # iou per object
    obj_eva = np.zeros((21,3))  # (tp, fp, fn)
    for i in range(H):
      for j in range(W):
        if y[i][j] > 20 or gt[i][j] >20:
          continue
        elif gt[i][j] == y[i][j]:
          # true positive
          obj_eva[gt[i][j]][0] += 1
        else:
          # false negative
          obj_eva[gt[i][j]][2] += 1
          # false positive
          obj_eva[y[i][j]][1] += 1
    # obj_iou = [round(any[0]/sum(any),2) if sum(any)!=0 else 0 for any in obj_eva]
    total = np.sum(obj_eva, axis=0)
    img_iou = round(total[0]/sum(total), 2)

    # draw mask on img
    img_color = image.copy()
    for i in np.unique(y):
      if i <= 21:  # exclude the boundary pixels
        img_color[y == i] = label_colours[i]
    disp = img_color.copy()
    cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)

    # draw mask on gt
    gt2 = Image.new("RGB",(W, H), 3) # the w, h are exchanged
    pixels = gt2.load()
    for i in range(H):
      for j in range(W):
        pixels[j,i] = label_colours[gt[i][j]]
    gt2 = np.array(gt2)

    # concatenate and add text
    out = np.concatenate([image/255, img_color/255, gt2/255, disp/255], axis=1)
    out_img = Image.fromarray((out * 255).astype(np.uint8))  #PIL does not accept float
    d = ImageDraw.Draw(out_img)
    fnt = ImageFont.truetype('/content/DeepLabV3_Plus-Tensorflow2.0/arial.ttf', 20)
    d.text((1800,15), "pixel-wise accuracy: "+str(p_a), font=fnt, fill = (255,255,255))
    d.text((1800,40), "image miou: "+str(img_iou), font=fnt, fill = (255,255,255))

    if save_img:
        out_img.save(save_dir+filename)
    else:
        plt.figure(figsize=(20, 10))
        out = np.array(out_img)
        plt.imshow(out)
    return p_a, obj_eva


def predict_label(model, img_path):
    img = img_to_array(load_img(img_path))
    img = cv2.resize(img, (W, H))
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pred_label = model.predict(img)
    pred_label = np.squeeze(pred_label)
    pred_label = np.argmax(pred_label, axis=2)
    return pred_label


def inference(img_lst, msk_lst, model, output):
    p_a_total = 0
    result_eva = np.zeros((21, 3))
    for i in tqdm(range(len(img_lst))):
        img = img_to_array(load_img(img_lst[i]))
        gt = np.array(Image.open(msk_lst[i]))
        p_a, obj_eva = pipeline(img, gt, model, filename=img_lst[i][-15:], save_dir=output, save_img=True)
        result_eva = np.add(result_eva, obj_eva)
        p_a_total += p_a
    print("pixelwise accuracy:", round(p_a_total / len(img_lst), 2))

    for i in range(21):
        if sum(result_eva[i]) == 0:
            print("class", i, "no object")
        else:
            print("class", i, "iou:", round(result_eva[i][0] / sum(result_eva[i]), 2))
    result_total = np.sum(result_eva, axis=0)
    print("mean iou:", round(result_total[0] / sum(result_total), 2))


def pixel_accuracy(pred, gt):
  total = 0
  correct = 0
  for i in range(H):
    for j in range(W):
      if gt[i][j] == pred[i][j]:
        correct+=1
      total+=1
  return round(correct/total, 2)

def main():
    FLAGS, unparsed = parser.parse_known_args()
    model = load_model(FLAGS.model)
    img_lst, msk_lst = create_list(FLAGS.img_txt, FLAGS.msk_txt)
    inference(img_lst, msk_lst, model, FLAGS.output)


if __name__ == '__main__':
    main()
