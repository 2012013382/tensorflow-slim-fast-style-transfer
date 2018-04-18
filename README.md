## tensorflow-slim-fast-style-transfer
Implement of fast-style-transfer by Tensorflow-slim(Easy to read)
## Requirements
Tensorflow(>=1.4)+slim+python(2.7)
## Dataset
COCO2014 http://images.cocodataset.org/zips/train2014.zip

You need unzip it, and place all images in 'train2014' folder.

All style images are provided by https://github.com/lengstrom/fast-style-transfer
## Parameters
All the parameters are the same with paper <Perceptual Losses for Real-Time Style Transfer and Super-Resolution>
  
STYLE_WEIGHT and CONTENT_WEIGHT are provided by https://github.com/hzy46/fast-neural-style-tensorflow 
## Usage
Please download slim vgg_16 check point from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

Then untar it and place it in 'model' folder.
```Bash
sudo ./convert_images_to_list.sh
```
to generate train images addresses list.
```Python
python train.py
```
It trains 'wave' style by default, you can change STYLE_IMAGE_PATH, TRAIN_CHECK_POINT and STYLE_WEIGHT in train.py to train other styles.

You can use 
```Bash
Tensorboard --logdir=.
```
in log folder to see more training details.

```Python
python test.py
```
for testing.
## Results
Please waiting...
## Reference
https://github.com/hzy46/fast-neural-style-tensorflow

https://github.com/lengstrom/fast-style-transfer
