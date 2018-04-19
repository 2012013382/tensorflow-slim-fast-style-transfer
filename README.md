## tensorflow-slim-fast-style-transfer
Implement of fast-style-transfer by Tensorflow-slim(Easy to read)
## Requirements
Tensorflow(>=1.4)+slim+python(2.7)
## Dataset
COCO2014 http://images.cocodataset.org/zips/train2014.zip

You need unzip it, and place all images in 'train2014' folder.

All style images are provided by https://github.com/lengstrom/fast-style-transfer
## Parameters
All the parameters are the same with paper Perceptual Losses for Real-Time Style Transfer and Super-Resolution
  
STYLE_WEIGHT and CONTENT_WEIGHT are provided by https://github.com/hzy46/fast-neural-style-tensorflow 
## Usage
Please download slim vgg_16 check point from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

Then untar it and place it in 'model' folder.
```Bash
sudo ./convert_images_to_list.sh train2014/
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

Model checkpoint I have trained: https://drive.google.com/drive/folders/1akw7PY6yF6A9hG_Au-j2U8-cDFfZ5uj8?usp=sharing

Please place style folders in 'model/trained_model/', and modify MODEL_PATH in test.py to get your own style images. For example, if you want to get 'scream' style images, please place 'scream' folder in 'model/trained_model/', and change MODEL_PATH in test.py into 'model/trained_model/scream/model.ckpt-20001'. Then place your test.jpg in 'test' folder and python test.py.
## Results
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/style_image.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/result1.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/result2.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/result3.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/result4.jpg)

Results with different steps.


![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/scream_bown_result.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/scream_building.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/candy_bown.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/candy_building.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/starry_bown.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/starry_building.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/wave_bown.jpg)
![](https://github.com/2012013382/tensorflow-slim-fast-style-transfer/blob/master/test/wave_building.jpg)
## Reference
https://github.com/hzy46/fast-neural-style-tensorflow

https://github.com/lengstrom/fast-style-transfer
