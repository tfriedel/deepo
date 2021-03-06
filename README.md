![deepo](https://user-images.githubusercontent.com/2270240/32102393-aecf573c-bb4e-11e7-811c-dc673cae7b9c.png)

[![CircleCI](https://img.shields.io/circleci/project/github/ufoym/deepo.svg)](https://circleci.com/gh/ufoym/deepo)
![license](https://img.shields.io/github/license/ufoym/deepo.svg)


***Deepo*** is a [*Docker*](http://www.docker.com/) image with a full reproducible deep learning research environment. It contains most popular deep learning frameworks:
[theano](http://deeplearning.net/software/theano),
[tensorflow](http://www.tensorflow.org),
[sonnet](https://github.com/deepmind/sonnet),
[pytorch](http://pytorch.org),
[keras](https://keras.io),
[lasagne](http://lasagne.readthedocs.io),
[mxnet](http://mxnet.incubator.apache.org),
[cntk](https://www.microsoft.com/en-us/cognitive-toolkit),
[chainer](https://chainer.org),
[caffe](http://caffe.berkeleyvision.org),
[torch](http://torch.ch/).

This is a fork of [ufoym/deepo](https://github.com/ufoym/deepo), built on python 3.6 instead of 3.5 and with some updated frameworks. 

- [Quick Start](#Quick-Start)
  - [Installation](#Installation)
  - [Usage](#Usage)
- [Comparison to Alternatives](#Comparison)
- [Licensing](#Licensing)

---

<a name="Quick-Start"/>

## Quick Start


<a name="Installation"/>

### Installation

#### Step 1. Install [Docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

#### Step 2. Obtain the Deepo image

You can either directly download the image from Docker Hub, or build the image yourself.

##### Option 1: Get the image from Docker Hub (recommended)
```bash
docker pull tfriedel/deepo
```
##### Option 2: Build the Docker image locally
```bash
git clone https://github.com/tfriedel/deepo.git
cd deepo && docker build -t tfriedel/deepo .
```
Note that this may take several hours as it compiles a few libraries from scratch.

<a name="Usage"/>

### Usage

Now you can try this command:
```bash
nvidia-docker run --rm tfriedel/deepo nvidia-smi
```
This should work and enables Deepo to use the GPU from inside a docker container.
If this does not work, search [the issues section on the nvidia-docker GitHub](https://github.com/NVIDIA/nvidia-docker/issues) -- many solutions are already documented. To get an interactive shell to a container that will not be automatically deleted after you exit do

```bash
nvidia-docker run -it tfriedel/deepo bash
```

If you want to share your data and configurations between the host (your machine or VM) and the container in which you are using Deepo, use the -v option, e.g.
```bash
nvidia-docker run -it -v /host/data:/data -v /host/config:/config tfriedel/deepo bash
```
This will make `/host/data` from the host visible as `/data` in the container, and `/host/config` as `/config`. Such isolation reduces the chances of your containerized experiments overwriting or using wrong data.


_You are now ready to begin your journey._


#### tensorflow
```$ python```
```python
>>> import tensorflow
>>> print(tensorflow.__name__, tensorflow.__version__)
tensorflow 1.4.0
```

#### sonnet
```$ python```
```python
>>> import sonnet
>>> print(sonnet.__name__, sonnet.__path__)
sonnet ['/usr/local/lib/python3.6/dist-packages/sonnet']
```

#### pytorch
```$ python```
```python
>>> import torch
>>> print(torch.__name__, torch.__version__)
torch 0.2.0_3
```

#### keras
```$ python```
```python
>>> import keras
>>> print(keras.__name__, keras.__version__)
keras 2.1.1
```

#### mxnet
```$ python```
```python
>>> import mxnet
>>> print(mxnet.__name__, mxnet.__version__)
mxnet 0.12.0
```

#### cntk
```$ python```
```python
>>> import cntk
>>> print(cntk.__name__, cntk.__version__)
cntk 2.2
```

#### chainer
```$ python```
```python
>>> import chainer
>>> print(chainer.__name__, chainer.__version__)
chainer 3.0.0
```

#### theano
```$ python```
```python
>>> import theano
>>> print(theano.__name__, theano.__version__)
theano 1.0.0rc1+11.ge47f1c9
```

#### lasagne
```$ python```
```python
>>> import lasagne
>>> print(lasagne.__name__, lasagne.__version__)
lasagne 0.2.dev1
```

#### caffe
```$ python```
```python
>>> import caffe
>>> print(caffe.__name__, caffe.__version__)
caffe 1.0.0
```

```$ caffe --version```
```
caffe version 1.0.0
```

#### torch
```$ th```
```
 │  ______             __   |  Torch7
 │ /_  __/__  ________/ /   |  Scientific computing for Lua.
 │  / / / _ \/ __/ __/ _ \  |  Type ? for help
 │ /_/  \___/_/  \__/_//_/  |  https://github.com/torch
 │                          |  http://torch.ch
 │
 │th>
```

<a name="Comparison"/>

## Comparison to alternatives
.                                                  | modern-deep-learning | dl-docker          | jupyter-deeplearning | Deepo
:------------------------------------------------: | :------------------: | :----------------: | :------------------: | :----------------:
 [ubuntu](https://www.ubuntu.com)                  | 16.04                | 14.04              | 14.04                | 16.04
 [cuda](https://developer.nvidia.com/cuda-zone)    | :x:                  | 8.0                | 6.5-8.0              | 8.0
 [cudnn](https://developer.nvidia.com/cudnn)       | :x:                  | v5                 | v2-5                 | v6
 [theano](http://deeplearning.net/software/theano) | :x:                  | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:
 [tensorflow](http://www.tensorflow.org)           | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:
 [sonnet](https://github.com/deepmind/sonnet)      | :x:                  | :x:                | :x:                  | :heavy_check_mark:
 [pytorch](http://pytorch.org)                     | :x:                  | :x:                | :x:                  | :heavy_check_mark:
 [keras](https://keras.io)                         | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:
 [lasagne](http://lasagne.readthedocs.io)          | :x:                  | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:
 [mxnet](http://mxnet.incubator.apache.org)        | :x:                  | :x:                | :x:                  | :heavy_check_mark:
 [cntk](http://cntk.ai)                            | :x:                  | :x:                | :x:                  | :heavy_check_mark:
 [chainer](https://chainer.org)                    | :x:                  | :x:                | :x:                  | :heavy_check_mark:
 [caffe](http://caffe.berkeleyvision.org)          | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:
 [torch](http://torch.ch/)                         | :x:                  | :heavy_check_mark: | :heavy_check_mark:   | :heavy_check_mark:

<a name="Licensing"/>

## Licensing

Deepo is [MIT licensed](https://github.com/tfriedel/deepo/blob/master/LICENSE).
