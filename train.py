import os
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from deeplab import DeepLabV3Plus
import keras as K

print('TensorFlow', tf.__version__)

parser = argparse.ArgumentParser()

parser.add_argument('--txt_dir', type=str,
                    default='/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_tensorflow_v1/dataset/',
                    help='directory that contains the train, val txt files.')
parser.add_argument('--ckpt_dir', type=str,
                    default="/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_plus_tensorflow_v2/checkpoints/training_1/cp-{epoch:04d}.ckpt",
                    help='directory that saves checkpoints.')
parser.add_argument('--tensorboard_dir', type=str,
                    default='/content/drive/My Drive/CS Internship/DeepLab_v3/deeplab_v3_plus_tensorflow_v2/logs',
                    help='directory that saves tensorboard logs.')
parser.add_argument('--restore', type=str,
                    default=None,
                    help='path of the checkpoint you want to restore.')
parser.add_argument('--epoch', type=int,
                    default=300,
                    help='number of epochs to train.')
parser.add_argument('--saving_interval', type=int,
                    default=2,
                    help='save every x epochs.')
parser.add_argument('--m', type=int,
                    default=0.9997,
                    help='training momentum.')
parser.add_argument('--e', type=int,
                    default=1e-5,
                    help='training epsilon.')
parser.add_argument('--lr', type=int,
                    default=1e-4,
                    help='learning rate.')

# Global variables
batch_size = 10
H, W = 512, 512
num_classes = 22  # including background and the boundary pixels
_DEPTH = 3
class_weights = [1] * 21 + [0]  # ignore the 22nd class


def make_list_from_txt(txt_dir):
    """"txt_dir: directory that contains the train, val txt files.
        return: lists of file paths of train/val image data."""
    f = open(txt_dir + 'train_img_full_path.txt', 'r')
    train_img_list = [line[:-1] for line in f.readlines()]
    f.close()

    f = open(txt_dir + 'train_msk_full_path.txt', 'r')
    train_msk_list = [line[:-1] for line in f.readlines()]
    f.close()

    f = open(txt_dir + 'val_img_full_path.txt', 'r')
    val_img_list = [line[:-1] for line in f.readlines()]
    f.close()

    f = open(txt_dir + 'val_msk_full_path.txt', 'r')
    val_msk_list = [line[:-1] for line in f.readlines()]
    f.close()

    assert len(train_img_list) == len(train_msk_list) and len(val_img_list) == len(val_msk_list)
    return train_img_list, train_msk_list, val_img_list, val_msk_list


def get_image(image_path, img_height=800, img_width=1600, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        img = tf.image.random_brightness(img, max_delta=50.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
            img_height, img_width]), dtype=tf.uint8)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
    return img


def random_crop(image, mask, H=512, W=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask


def load_data(image_path, mask_path, H=512, W=512):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(
        mask_path, mask=True, flip=flip)
    image, mask = random_crop(image, mask, H=H, W=W)
    return image, mask


def make_dataset(train_img_list, train_msk_list, val_img_list, val_msk_list):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img_list, train_msk_list))
    train_dataset = train_dataset.shuffle(buffer_size=128)
    train_dataset = train_dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_data,
                                           batch_size=batch_size,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_img_list,
                                                      val_msk_list))
    val_dataset = val_dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_data,
                                           batch_size=batch_size,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def weightedLoss(originalLossFunc, weightsList):  # function to set weights on loss function
    def lossFunc(true, pred):
        axis = -1  # if channels last
        # axis=  1 #if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)
        # if your loss is sparse, use only true as classSelectors

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]


        # casting boolean to float for calculations
        # each tensor in the list contains 1 where ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        # for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(classSelectors, weightsList)]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        # make sure your originalLossFunc only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true, pred)
        loss = loss * weightMultiplier

        return loss

    return lossFunc


def define_model(H, W, num_classes, momentum=0.9997, epsilon=1e-5, learning_rate=1e-4):
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = weightedLoss(loss, class_weights)  # use the weighed loss function
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DeepLabV3Plus(H, W, num_classes)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = momentum
                layer.epsilon = epsilon
            elif isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

        model.compile(loss=loss,
                      optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
    return model


def define_callbacks(tb_logs_path, checkpoint_path, saving_interval=2):
    tb = TensorBoard(log_dir=tb_logs_path, write_graph=True, update_freq='batch')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    mc = ModelCheckpoint(mode='min',
                         filepath=checkpoint_path,
                         monitor='val_loss',
                         save_weights_only='True',
                         period=saving_interval,  # save the checkpoint every x epochs
                         verbose=1)
    return [mc, tb]


def main():
    FLAGS, unparsed = parser.parse_known_args()
    train_img_list, train_msk_list, val_img_list, val_msk_list = make_list_from_txt(FLAGS.txt_dir)
    print("Successfully made data lists!")
    train_dataset, val_dataset = make_dataset(train_img_list, train_msk_list, val_img_list, val_msk_list)
    print("Successfully made dataset!")
    momentum, epsilon, learning_rate = FLAGS.m, FLAGS.e, FLAGS.lr
    model = define_model(H, W, num_classes, momentum, epsilon, learning_rate)
    print("Successfully defined the model!")
    callbacks = define_callbacks(FLAGS.tensorboard_dir, FLAGS.ckpt_dir, FLAGS.saving_interval)
    if FLAGS.restore:  # the restore flag is not None
        print("Restore training weights...")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model.load_weight(FLAGS.restore)

    print("Start training...")
    model.fit(train_dataset,
              steps_per_epoch=len(train_img_list) // batch_size,
              epochs=FLAGS.epoch,
              verbose=1,
              validation_data=val_dataset,
              validation_steps=len(val_img_list) // batch_size,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
