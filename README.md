[![HitCount](http://hits.dwyl.io/srihari-humbarwadi/https://githubcom/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow20.svg)](http://hits.dwyl.io/srihari-humbarwadi/https://githubcom/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow20)


## Architecture
![model](deeplabv3plus.png)

## Train the model
Check the flags
```bash
python3 train.py -h
```

## Trained weights
[trained weights](https://drive.google.com/open?id=1wRXyIGUVRws3BJHX-UrNDSZGDzUzgVMx)


## To Do
- [x] train on PASCAL 2012 
- [ ] try other tf strategies
- [x] provide a better restoring function
- [ ] test the performance on different classes
- [ ] handle the boundary pixels in another way and compare the performance
- [ ] use TFRecord to accelerate the training process
- [ ] solve the unknown error at the beginning of each new training


