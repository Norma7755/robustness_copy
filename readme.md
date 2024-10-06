# Robustness of SSM

Code for NIPS 2024 paper [Exploring Adversarial Robustness of Deep State
Space Models](https://arxiv.org/pdf/2406.05532)


## Install Mamba 

Install Mamba (mamba-ssm, causal-conv1d) following the original author's link: https://github.com/state-spaces/mamba


## Dependencies

```shell
pip install -r requirements.txt
```

## Dataset Preparation

1. For MNIST and CIFAR10 datasets, the script automatically downloads.

2. For Tiny-ImageNet dataset refer to the following link: https://github.com/tjmoon0104/pytorch-tiny-imagenet


## Quick Start

All run scripts are located in the `scripts` folder. To obtain the training results of various models on the MNIST dataset under Standard Training, Madry Training, and Trades Training methods, please run the following command:

```python
python train_trades_mnist.py --AT_type Nat --model_name SSM --attack_type PGD --model-dir checkpoints/model-MNIST
```

Where `--AT_type` specifies different training methods, `--model_name` specifies the model type, and `--model-dir` specifies the location to save the results.

To obtain results for CIFAR10 and Tiny-Imagenet, please run the `train_trades_cifar10.py` and `train_freeat_tinyimagenet.py` scripts:

```python
python train_trades_cifar10.py --AT_type Nat --model_name SSM --attack_type PGD --model-dir checkpoints/model-CIFAR10
python train_trades_tinyimagenet.py --AT_type Nat --model_name SSM --attack_type PGD --model-dir checkpoints/model-tinyimagenet
```

To obtain the training results of various models under the FreeAT and YOPO training methods, please run the following commands:

```python
python train_freeat_mnist.py --AT_type FreeAT --model_name SSM --attack_type PGD --model-dir checkpoints/model-MNIST
python train_yopo_mnist.py --AT_type YOPO --model_name SSM --attack_type PGD --model-dir checkpoints/model-MNIST
```

To obtain the results of the Adss method, add the Adss method parameters `--use_inject --inject_method 1` to the SSM and DSS models. The specific commands are as follows:

```python
python train_freeat_mnist.py --AT_type FreeAT --model_name SSM --attack_type PGD --model-dir checkpoints/model-MNIST --use_inject --inject_method 1
python train_freeat_mnist.py --AT_type FreeAT --model_name DSS --attack_type PGD --model-dir checkpoints/model-MNIST --use_inject --inject_method 1
```

The specific implementation of the Adss module is located in the file `models/src/models/sequence/ss/s4.py`.


## Contact
Please contact us or post an issue if you have any questions.

* Biqing Qi (qibiqing7@gmail.com)
* Yiang Luo (normanluo668@gmail.com)
* Junqi Gao (gjunqi97@gmail.com)
* Pengfei Li (lipengfei0208@gmail.com)


## Citation
```BibTeX
@article{qi2024exploring,
  title={Exploring Adversarial Robustness of Deep State Space Models},
  author={Qi, Biqing and Luo, Yang and Gao, Junqi and Li, Pengfei and Tian, Kai and Ma, Zhiyuan and Zhou, Bowen},
  journal={arXiv preprint arXiv:2406.05532},
  year={2024}
}
```