# AgileNN: Real-time neural network inference on extremely weak devices: agile offloading with explainable AI (MobiCom'22)

## Introduction
This code repository provides training scripts of AgileNN. It leverages eXplainable AI (XAI) to enable fast and accurate neural network inference on extremely weak devices (e.g., MCUs) with wireless connections. It shifts the rationale of offloading from fixed to data-centric and agile. Please refer to our [paper](https://dl.acm.org/doi/10.1145/3495243.3560551) for details. 

## Requirements
* Python 3.7+
* [tensorflow 2](https://www.tensorflow.org/install)
* [tensorflow-datasets](https://github.com/tensorflow/datasets)
* [tensorflow-addons](https://github.com/tensorflow/addons)
* [tiny-imagenet-tfds](https://github.com/ksachdeva/tiny-imagenet-tfds)
* [tqdm](https://github.com/tqdm/tqdm)

## How to use
Below shows an example of training on CIFAR100 dataset:

At the bottom of `train_evaluator.py`, uncomment the following.
```
train_effnetv2_on_cifar100('effnetv2_pretrained', 'logs')
```
Run `train_evaluator.py` to train Reference NN on CIFAR100. The trained Reference NN and its log will be saved to `saved_models/` and `logs/` respectively.

In `main.py`, change the model path to the trained Reference NN. For example,
```
EVALUATOR_PATH = 'saved_models/effnetv2_pretrained_x1886.tf'
``` 
Configure other hyperparameters in `main.py`. Some parameters such as `LAMBDA` and `NUM_CENTROIDS` may need finetuning for the best outcome.
```python
# training config of AgileNN
DATASET = 'cifar100' # dataset to be trained on, selected from ['cifar10', 'cifar100', 'svhn', 'imagenet200']
EVALUATOR_PATH = 'saved_models/effnetv2_pretrained_x1886.tf'
SPLIT_RATIO = 0.2 # num_local_features / (num_local_features + num_remote_features)
RHO = 0.8 # skewness of feature importance
LAMBDA = 0.8 # to balance loss terms, lambda * L_ce + (1 - lambda) (L_skewness + L_disorder)
NUM_CENTROIDS = 8 # quantize remote features to log2(NUM_CENTROIDS) bit representation 
```
Run `main.py`. It should achieve ~73% final accuracy and ~83% skewness. Computing gradients on gradients can be expensive, you may need powerful GPUs with large memory.

The trained modules of AgileNN can be converted to [tflite](https://www.tensorflow.org/lite) or [tflite-micro](https://www.tensorflow.org/lite/microcontrollers) format for deployment. The quantized remote features can be further compressed by lossless algorithms, e.g., [LZW](https://rosettacode.org/wiki/LZW_compression#Python) and [Huffman](https://rosettacode.org/wiki/Huffman_coding#C++).

## Citation
```
@inproceedings{huang2022real,
  title={Real-time neural network inference on extremely weak devices: agile offloading with explainable AI},
  author={Huang, Kai and Gao, Wei},
  booktitle={Proceedings of the 28th Annual International Conference on Mobile Computing And Networking},
  pages={200--213},
  year={2022}
}
```

