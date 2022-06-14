# Making NLP Models more Interpretable using Deep k-Nearest Neighbors

Original paper: Papernot, Nicolas, and Patrick McDaniel. [Deep k-nearest neighbors: Towards confident, interpretable and robust deep learning.](https://arxiv.org/abs/1803.04765)

Prior NLP implementation: Eric Wallace, Shi Feng, and Jordan Boyd-Graber. 2018. [Interpreting Neural Networks with Nearest Neighbors. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP](https://aclanthology.org/W18-5416): Analyzing and Interpreting Neural Networks for NLP, pages 136–144.

This repository currently is an re-implementation of the pseudo-code for DkNN proposed by Papernot et. al. 

# Dataset

We derive our test data from Giovanni Da San Martino, Seunghak Yu, Alberto Barrón-Cedeño, Rostislav Petrov, and Preslav Nakov. 2019. [Fine-grained analysis of propaganda in news article](https://aclanthology.org/D19-1565.pdf). In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5636–5646, Hong Kong, China. Association for Computational Linguistics.

The actual dataset is found [here](https://propaganda.qcri.org/). The user may specify a local directory in which data may be found, where we place them under `/data/protechn_corpus_eval/` (same organizational structure as the direct download link).

Each data path (e.g. `/data/protechn_corpus_eval/train`) consist of multiple pairs of '.tsv' and '.txt' files grouped by an unique ID per pair. Specifically, the dataset is a collection of article snippets in the format of:

`article<ID>.labels.tsv`: a tab-separated file with the columns - articleID, Propaganda Technique (one of 18 from Martino et. al.), Span Start Index, Span End Index.

and

`article<ID>.txt`: an article snippet which may span several lines and/or paragraphs

# Baseline

# Installation

```
conda create --name sam python=3.9
pip3 install git+https://github.com/huggingface/transformers
```