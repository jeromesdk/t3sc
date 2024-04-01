# A Trainable Spectral-Spatial Sparse Coding Model for Hyperspectral Image Restoration

The official PyTorch implementation of the paper _A Trainable Spectral-Spatial Sparse Coding Model for Hyperspectral Image Restoration_ (Neurips 2021) is available at https://github.com/inria-thoth/T3SC/tree/main. However, we encountered several issues while runing it, the purpose of this project was to debug those errors.

[`[arxiv]`](https://arxiv.org/abs/2111.09708)

![](figs/architecture.png)


## Installation

Developped with Python 3.8.8. We ran the code on a RTX3060. Please, visit the official pytorch website to see the version of pytorch to install: https://pytorch.org/
```
$ git clone https://github.com/inria-thoth/T3SC
$ cd t3sc && pip install -r requirements.txt
```

## Training


To launch a training:
```
$ python main.py data={icvl,dcmall} noise={constant,uniform,correlated,stripes} [+noise-specific params]
```
Data should be downloaded automatically to `data/ICVL` or `data/DCMall` if it is not there already.

**NOTE**: For uniform and stripes noises, better results are obtained with Noise Adaptive Sparse Coding.
To enable this feature, use `model.beta=1` for both training and testing.

### Examples

Runing takes a very long time. We highly recommand to download the trained mmodel so that you don't need to train it. If you still want to train it, set the `num_workers` according to your configuration in the file `dcmall.yaml` or `icvl.yaml` located in `t3sc/config/data`

ICVL dataset with constant gaussian noise:
```
$ python main.py data=icvl noise=constant noise.params.sigma=50
```

ICVL dataset with stripes noise:
```
$ python main.py data=icvl noise=stripes
```


## Test

To test from a checkpoint:
```
$ python main.py mode=test data={icvl,dcmall} noise={constant,uniform,correlated,stripes} [+noise-specific params] model.ckpt=path/to/ckpt
```

You can also have the 95% confidence interval of the metrics by executing the statistic python file:

```
$ python statistics.py
```

Some pre-trained models can be found [here](http://pascal.inrialpes.fr/data2/tbodrito/t3sc/). Two pretrained models are already presents, you can use them.

### Example
To test ICVL with constant noise:
```
$ python main.py mode=test data=icvl noise=constant noise.params.sigma=50 model.ckpt=path/to/icvl_constant_50.ckpt
```

## Slides of the presentation

The presentation of the project is available, check the file `computational imaging.pdf`.
