# AgileNN: Real-time neural network inference on extremely weak devices: agile offloading with explainable AI (MobiCom'22)

## Introduction
This code repository provides training scripts of AgileNN. It leverages eXplainable AI (XAI) to enable fast and accurate neural network inference on extremely weak devices (e.g., MCUs) with wireless connections. It shifts the rationale of offloading from fixed to data-centric and agile. Please refer to our [paper](https://dl.acm.org/doi/10.1145/3495243.3560551) for details. 

## Requirements
* Python 3.7+
* [tensorflow 2](https://www.tensorflow.org/install)
* [tensorflow-datasets](https://github.com/tensorflow/datasets)
* [tensorflow-addons](https://github.com/tensorflow/addons)
* [tiny-imagenet-tfds](https://github.com/ksachdeva/tiny-imagenet-tfds)
* tqdm

## Usage
Below shows an example of training AgileNN on CIFAR100 dataset. First pre-train Reference NN on CIFAR100 dataset:

```
python train_evaluator.py --dataset cifar100
```
Then train AgileNN on CIFAR100 dataset:
```
python main.py --dataset cifar100 --split_ratio 0.2 --rho 0.8 --klambda 0.8 --num_centroids 8
```

It should achieve ~73% final accuracy and ~83% skewness. Computing gradients on gradients can be expensive, so you may need powerful GPUs with large memory.

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

