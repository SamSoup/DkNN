# Making NLP Models more Interpretable using Deep k-Nearest Neighbors

Original paper: Papernot, Nicolas, and Patrick McDaniel. [Deep k-nearest neighbors: Towards confident, interpretable and robust deep learning.](https://arxiv.org/abs/1803.04765)

Prior NLP implementation: Eric Wallace, Shi Feng, and Jordan Boyd-Graber. 2018. [Interpreting Neural Networks with Nearest Neighbors. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP](https://aclanthology.org/W18-5416): Analyzing and Interpreting Neural Networks for NLP, pages 136–144.

This repository currently is an re-implementation of the pseudo-code for DkNN proposed by Papernot et. al. 

# Dataset

We use the dataset from Antigoni Maria Founta et al. [Large scale crowdsourcing and characterization of twitter abusive behavior](https://arxiv.org/pdf/1802.00393.pdf). In: Twelfth International AAAI Conference on Web and Social Media. 2018. The complete dataset citation is as belows:

```
Antigoni-Maria Founta; Constantinos Djouvas; Despoina Chatzakou; Ilias Leontiadis; Jeremy Blackburn; Gianluca Stringhini; Athena Vakali; Michael Sirivianos; Nicolas Kourtellis, 2018, "Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior", doi:10.5072/FK2/ZDTEMN, Root, V1, UNF:6:Gis9lCfXBUc7fdE0YoydlA== 
```

The dataset covers 80 thousand tweets categorized under hateful, abusive, normal, or spam. We do not use any additional metadata besides the crowdsourced majority vote label nor retweet information.

# Baseline

BERT

BART

# Installation

```
conda create --name sam python=3.9
conda activate sam
pip3 install git+https://github.com/huggingface/transformers
conda install -c intel scikit-learn
```
