# CV-Transformer

## Introduction

This Git repository provides the code for a complex-valued transformer architecture. The key modules can be found in the folder 'attention'. 

Additionally it provides the experiments presented in the Paper 'Building Block for a complex-valued transformer architecture' by Florian Eilers and Xiaoyi Jiang.

## How to use CV-Transformer
The module is written in pytorch. The complex-valued transformer has the same modular structure and the same parameters as the real-valued transformer implementation from pytorch: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

It can be used in any pytorch architecture in the same way, as the real-valued module. The only limitation is, that it expects complex-valued inputs. If one wants to use it with real-valued input, those can be transformed to the complex domain with pytorch's complex constructor: https://pytorch.org/docs/stable/generated/torch.complex.html with an empty (0) imaginary part.

## How to reproduce experiments from the Paper

To reproduce the experiments from the paper, the Musicnet dataset needs to be downloaded: https://zenodo.org/record/5120004#.Yhxr0-jMJBA

Then the preprocessing pipeline in the preproccesing folder has to be run. 
After the preprocessing, the path to the dataset has to be inserted into the respective execute...py script. All parameters are set as in the experiments in the paper but can be changed there as well. To run the complex-valued versions, add the parameter --sm_variant. Options are realip, absip, absonlyip, naivip, realcp, abscp, absonlycp, naivcp. 

## Dependencies:
All Dependencies can be found in the conda.yml which can be used with Conda to directly install the desired environment. The core dependencies will be added here later as well.

## Usage
This repository is free for use and modification for scientific users. If you use our work please cite:
...

