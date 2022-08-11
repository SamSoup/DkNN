# Making NLP Models more Interpretable using Deep k-Nearest Neighbors

Original paper: Papernot, Nicolas, and Patrick McDaniel. [Deep k-nearest neighbors: Towards confident, interpretable and robust deep learning.](https://arxiv.org/abs/1803.04765)

Prior NLP implementation: Eric Wallace, Shi Feng, and Jordan Boyd-Graber. 2018. [Interpreting Neural Networks with Nearest Neighbors. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP](https://aclanthology.org/W18-5416): Analyzing and Interpreting Neural Networks for NLP, pages 136–144.

This repository currently is an re-implementation of the pseudo-code for DkNN proposed by Papernot et. al. 

# Dataset

## Founta et. al.

We use the dataset from Antigoni Maria Founta et al. [Large scale crowdsourcing and characterization of twitter abusive behavior](https://arxiv.org/pdf/1802.00393.pdf). In: Twelfth International AAAI Conference on Web and Social Media. 2018. The complete dataset citation is as belows:

```
Antigoni-Maria Founta; Constantinos Djouvas; Despoina Chatzakou; Ilias Leontiadis; Jeremy Blackburn; Gianluca Stringhini; Athena Vakali; Michael Sirivianos; Nicolas Kourtellis, 2018, "Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior", doi:10.5072/FK2/ZDTEMN, Root, V1, UNF:6:Gis9lCfXBUc7fdE0YoydlA== 
```

The dataset covers 80 thousand tweets categorized under hateful, abusive, normal, or spam. We do not use any additional metadata besides the crowdsourced majority vote label nor retweet information. We binarize the dataset into tweets of hate vs non-hate (abusive + normal + spam), see `preprocess.py`. Because the longest tweet is 964 characters, we pad everything to 1024 characters long.

## Toxigen

250k English samples about implicit toxic sentences.

```
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

# Baseline

DistillBert
DistillRoBERTa
Bart-base

# Installation

```
conda create --name DkNN python=3.9
conda activate DkNN
conda install -n DkNN ipykernel --update-deps --force-reinstall
conda install -c huggingface transformers
conda install -c conda-forge ray-tune datasets scikit-learn
python -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install pyLSHash
```

On TACC: add these flags to pip install

--user 
--no-cache-dir

idev -p gpu-a100

# Execution

Note that it's the responsibility of the user to specify which layers' representations to save for DkNN using the `layers_to_save` argument. 

```
python3 main.py <path_to_json_configuration_file>
```

# Documentation

