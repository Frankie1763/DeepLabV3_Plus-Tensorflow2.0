# A file contraining all the MACROS and FLAGS

FLAGS = {
    "model_dir": "../model/model1/cp-{epoch:04d}.ckpt",
    "tb_dir": "../tensorboard/model1",
    "restore": None,
    "train_epochs": 50,
    "saving_interval": 5,
    "momentum": 0.9,
    "epsilon": 1e-5,
    "lr": 1e-4,
    "decay": 1e-6,
    "starting_epoch": 1,
    "backbone": "resnet101",
    "batch_size": 16,
    "data_dir": "../dataset/"
}

HEIGHT, WIDTH = 513, 513
num_classes = 21
DEPTH = 3
MIN_SCALE = 0.5
MAX_SCALE = 2.0
IGNORE_LABEL = 255

NUM_IMAGES = {
    'train': 10582,
    'val': 1449,
}
