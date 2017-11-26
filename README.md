# Matrix-CapsNet-EM-routing-tensorflow
This is a trial implementation of Hinton group's [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb) in TensorFlow and Python programming language. （仅供交流学习使用 Education purpose only）

# Project completeness 10%, models are forked from CapsNet implementation.

## Related Work

Link to CapsNet implementation （仅供交流学习使用 Education purpose only）: [https://github.com/yhyu13/CapsNet-python-tensorflow](https://github.com/yhyu13/CapsNet-python-tensorflow)

## TODO

- [x] Convert smallNORB dataset into TFRecords
- [x] Split smallNORB dataset into manageable chunks (optional)
- [ ] Implement Convolutional Capsule Layers
- [ ] Implement EM routing
- [ ] Implement Spread Loss with linear annealing
- [ ] Reproduce test error on smallNORB dataset (if manageable)
- [ ] Reproduce experiments on smallNORB dataset (if manageable)

## Dataset

[smallNORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)

> This database is intended for experiments in 3D object reocgnition from shape. It contains images of 50 toys belonging to 5 generic categories: four-legged animals, human figures, airplanes, trucks, and cars. The objects were imaged by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees every 5 degrees), and 18 azimuths (0 to 340 every 20 degrees).

> The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9), and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5).

### Download NORB

```./data/download.sh```

![](/figure/sample1.png)

![](/figure/sample2.png)

### Write to TFRecord

```python dataset.py tfrecord```

## Reference:

[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)

[MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb)

Y. LeCun, F.J. Huang, L. Bottou, Learning Methods for Generic Object Recognition with Invariance to Pose and Lighting. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004
