import tensorflow as tf
# import argparse
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from flags import *
from utils import preprocessing
import os

print('TensorFlow', tf.__version__)


# parser = argparse.ArgumentParser()

# parser.add_argument('--restore', type=str,
#                     default=None,
#                     help='path of the checkpoint you want to restore.')

def get_filenames(is_training, data_dir):
    """Return a list of filenames.
    Args:
	is_training: A boolean denoting whether the input is for training.
	data_dir: path to the the directory containing the input data.
	Returns:
	A list of file names.
    """
    if is_training:
        return [os.path.join(data_dir, 'voc_train.record')]
    else:
        return [os.path.join(data_dir, 'voc_val.record')]


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
        'image/height':
            tf.io.FixedLenFeature((), tf.int64),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
            tf.io.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed = tf.io.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape=[]), DEPTH)
    image = tf.cast(tf.image.convert_image_dtype(image, dtype=tf.uint8), tf.float32)
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.cast(tf.image.convert_image_dtype(label, dtype=tf.uint8), tf.int32)
    label.set_shape([None, None, 1])

    return image, label


def preprocess_image(image, label, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Randomly scale the image and label.
        image, label = preprocessing.random_rescale_image_and_label(image, label, MIN_SCALE, MAX_SCALE)

        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image, label = preprocessing.random_crop_or_pad_image_and_label(image, label, HEIGHT, WIDTH, IGNORE_LABEL)

        # Randomly flip the image and label horizontally.
        image, label = preprocessing.random_flip_left_right_image_and_label(image, label)

        image.set_shape([HEIGHT, WIDTH, 3])
        label.set_shape([HEIGHT, WIDTH, 1])

    image = preprocessing.mean_image_subtraction(image)

    return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.
    Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: The directory containing the input data.
        batch_size: The number of samples per batch.
        num_epochs: The number of epochs to repeat the dataset.

    Returns:
        A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)  # flat_map make sure: order of the dataset stays the same

    if is_training:
        # choose shuffle buffer sizes, larger sizes result in better randomness, smaller sizes have better performance.
        # Pascal is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=NUM_IMAGES['train'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, is_training))
    dataset = dataset.prefetch(batch_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    # iterator = dataset.make_initializable_iterator() # dataset.make_one_shot_iterator()
    # images, labels = iterator.get_next()
    # iterator_init_op = iterator.initializer

    return dataset


class MyWeightedLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    def call(self, y_true, y_pred, sample_weight=None):
        y_true_flat = tf.reshape(y_true, [-1, ])
        valid_indices = tf.cast(y_true_flat <= num_classes - 1, tf.int32)
        valid_labels = tf.dynamic_partition(y_true_flat, valid_indices, num_partitions=2)[1]

        # get valid logits
        logits_by_num_classes = tf.reshape(y_pred, [-1, num_classes])
        valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]

        return super().call(valid_labels, valid_logits)


# define MyMeanIOU to use argmax to preprocess the result
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_flat = tf.reshape(y_true, [-1, ])
        valid_indices = tf.compat.v1.to_int32(y_true_flat <= num_classes - 1)
        valid_labels = tf.dynamic_partition(y_true_flat, valid_indices, num_partitions=2)[1]

        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred_flat = tf.reshape(y_pred, [-1, ])
        valid_preds = tf.dynamic_partition(y_pred_flat, valid_indices, num_partitions=2)[1]
        return super().update_state(valid_labels, valid_preds)


def define_model(backbone, H, W, num_classes, momentum=0.9997, epsilon=1e-5, learning_rate=1e-2, decay=1e-6):
    if backbone == "resnet50":
        from deeplab_resnet50 import DeepLabV3Plus
    elif backbone == "resnet101":
        from deeplab_resnet101 import DeepLabV3Plus
    elif backbone == "xception":
        from deeplab_xception import DeepLabV3Plus
    elif backbone == "renet50_duc":
        from deeplab_resnet50_duc import DeepLabV3Plus
    # loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = weightedLoss(loss, class_weights)  # use the weighed loss function
    loss = MyWeightedLoss(from_logits=True)
    model = DeepLabV3Plus(H, W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = momentum
            layer.epsilon = epsilon
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate, decay=decay),
                  metrics=[MyMeanIOU(num_classes=21)])
    return model


def define_callbacks(tb_logs_path, checkpoint_path, saving_interval=2):
    tb = TensorBoard(log_dir=tb_logs_path, write_graph=True, update_freq='batch')
    mc = ModelCheckpoint(mode='min',
                         filepath=checkpoint_path,
                         monitor='val_loss',
                         save_weights_only='True',
                         period=saving_interval,  # save the checkpoint every x epochs
                         verbose=1)
    return [mc, tb]


def main():
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train_dataset = input_fn(True, FLAGS['data_dir'], FLAGS['batch_size'], FLAGS['train_epochs'])
    val_dataset = input_fn(False, FLAGS['data_dir'], 1, num_epochs=1)

    momentum, epsilon, learning_rate, decay = FLAGS['momentum'], FLAGS['epsilon'], FLAGS['lr'], FLAGS['decay']
    model = define_model(FLAGS['backbone'], HEIGHT, WIDTH, num_classes, momentum, epsilon, learning_rate, decay)
    print("Successfully defined the model!")
    callbacks = define_callbacks(FLAGS['tb_dir'], FLAGS['model_dir'], FLAGS['saving_interval'])
    if FLAGS['restore']:  # the restore flag is not None
        print("Restore training weights...")
        model.load_weights(FLAGS['restore'])

    print("Start training...")
    starting_epoch = FLAGS['starting_epoch']
    model.fit(train_dataset,
              steps_per_epoch=NUM_IMAGES['train'] // FLAGS['batch_size'],
              epochs=FLAGS['train_epochs'],
              verbose=1,
              validation_data=val_dataset,
              validation_steps=NUM_IMAGES['val'] // FLAGS['batch_size'],
              initial_epoch=starting_epoch,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
