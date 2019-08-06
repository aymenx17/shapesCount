# ShapeCount
Counting the number of shapes per geometric type


### Intro

Implement an algorithm to recognize and count the number of occurrence of three different shapes
 (i.e. square, circle, triangle) in a given image. To this purpose there are different methods someone could apply
 (e.g. hard-coded logic, image processing heuristics, shallow machine learning, deep learning, etc).

Here we present a deep learning based solution that well generalizes on the test set. The dataset consists of 5000 annotated examples:
  -  JPG image of size 500 x 500 with arbitrary numbers of squares, circles, and triangles.
  -  Text file stating how many squares, circles, and triangles are in the image.

![](https://github.com/aymenx17/shapesCount/blob/master/project_imgs/shapesCount.png)

### Environment

The code works and has been tested with the following packages version:
- Pytorch 1.0
- Python 3.6.6
- opencv 3.4.1
- numpy 1.16.2

For a more comprehensive list of required packages you may refer to the file ‘​env.yaml​’.
You may also use conda package manager to rebuild the specific working environment:


```python
conda env create -f env.yaml
```

Note: I worked on a remote machine along with Pycharm configured on Paperspace, a cloud computing which provides
virtual machines with already installed all the GPU libraries commonly required by deep learning frameworks.

### Run Code

```python

# training
python train.py

# demo
python demo.py -m checkpoints/net_6.pth -i path_to_image
python demo.py -m checkpoints/net_6.pth -f path_to_folder_of_images

```


#### Sample Visualizations

![](https://github.com/aymenx17/shapesCount/blob/master/project_imgs/sample_1.png)
![](https://github.com/aymenx17/shapesCount/blob/master/project_imgs/sample_3.png)
![](https://github.com/aymenx17/shapesCount/blob/master/project_imgs/sample_4.png)

The original input image followed by three attention maps, corresponding to the three type of shapes;
in order: squares, circles, triangles.

#### Test Accuracy

The algorithm has been evaluated on 500 images of test set.
The following metrics will be measured:
  - Prediction accuracy shape-wise: accuracy over total number of successfully counted shapes in the evaluation set.
  - Prediction accuracy image-wise: accuracy over total number of images where all shapes were successfully counted.

![](https://github.com/aymenx17/shapesCount/blob/master/project_imgs/shapesCount_accuracy.png)
