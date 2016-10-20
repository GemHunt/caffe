---
title: CaffeNet C++ Classification example
description: A simple example performing image classification using the low-level C++ API.
category: example
include_in_docs: true
priority: 10
---

# Notes from Paul: Currently working on adding lmdb classification to this example...


Classification_CPP:
* Test my own lmdbs to see if the keys pass through. 
* Change ForwardPrefilled to Forward 
* I don’t need large batches, so use this for a while before doing anything else. 

Post on Google caffe users asking for help: 
* I am looking for commentary on this. Why hasn’t anyone done this yet? Is there a better way?

* For reading the lmdb I just converted the datum to cv:Mat, but I am guessing this does not need to be done. It’s nice because it uses the same code. 
* Add reading image names from a text file
* Add batching. On the cpu it doesn’t matter, but on the GPU this currently just fails when the GPU memory is used up. 
* Use multiple classification threads to ensure the GPU is always fully utilized and not waiting for an I/O blocked CPU thread.
* Understand why objects are past with const and some are not. 
* Move most of code left main into the Classifier class
* Go back to the way the code was written now that I am making CV:Mat for imdbs?
* Allow Multi GPU
* Work on the class interface (Make more of the functions private.)
* Add a parameter for top results. (5 is hardcoded right now) 
* The readme needs to be updated. 

My Personal Learning:
* I want to nitpick the C++ code before and after. 
* I need to have a more solid understanding on how variables are being passed into functions here. 
* How do you find the implementation and declaration of variables in a terminal?








# Classifying ImageNet: using the C++ API

Caffe, at its core, is written in C++. It is possible to use the C++
API of Caffe to implement an image classification application similar
to the Python code presented in one of the Notebook examples. To look
at a more general-purpose example of the Caffe C++ API, you should
study the source code of the command line tool `caffe` in `tools/caffe.cpp`.

## Presentation

A simple C++ code is proposed in
`examples/cpp_classification/classification.cpp`. For the sake of
simplicity, this example does not support oversampling of a single
sample nor batching of multiple independent samples. This example is
not trying to reach the maximum possible classification throughput on
a system, but special care was given to avoid unnecessary
pessimization while keeping the code readable.

## Compiling

The C++ example is built automatically when compiling Caffe. To
compile Caffe you should follow the documented instructions. The
classification example will be built as `examples/classification.bin`
in your build directory.

## Usage

To use the pre-trained CaffeNet model with the classification example,
you need to download it from the "Model Zoo" using the following
script:
```
./scripts/download_model_binary.py models/bvlc_reference_caffenet
```
The ImageNet labels file (also called the *synset file*) is also
required in order to map a prediction to the name of the class:
```
./data/ilsvrc12/get_ilsvrc_aux.sh
```
Using the files that were downloaded, we can classify the provided cat
image (`examples/images/cat.jpg`) using this command:
```
./build/examples/cpp_classification/classification.bin \
  models/bvlc_reference_caffenet/deploy.prototxt \
  models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  examples/images/cat.jpg
```
The output should look like this:
```
---------- Prediction for examples/images/cat.jpg ----------
0.3134 - "n02123045 tabby, tabby cat"
0.2380 - "n02123159 tiger cat"
0.1235 - "n02124075 Egyptian cat"
0.1003 - "n02119022 red fox, Vulpes vulpes"
0.0715 - "n02127052 lynx, catamount"
```

## Improving Performance

To further improve performance, you will need to leverage the GPU
more, here are some guidelines:

* Move the data on the GPU early and perform all preprocessing
operations there.
* If you have many images to classify simultaneously, you should use
batching (independent images are classified in a single forward pass).
* Use multiple classification threads to ensure the GPU is always fully
utilized and not waiting for an I/O blocked CPU thread.
