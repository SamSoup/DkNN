from typing import List, Tuple
import numpy as np

def get_all_labels(labels_file: str, train_dataset, eval_dataset, test_dataset) -> List[str]:
    """
    Obtain the list of labels either from a given file or by computing the 
    unique labels across train, eval, and test dataset
    """
    if labels_file is not None:
        with open("./data/protechn_corpus_eval/propaganda-techniques-names.txt") as f:
            return f.read().splitlines()
    else:
        pass

def parse_label(label_path: str) -> List[Tuple[int, int, str, int, int]]:
    """
    
    """
    labels = []
    with open(label_path) as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            labels.append([int(parts[2]), int(parts[3]), parts[1], 0, 0])
            labels = sorted(labels) 

    if labels:
        length = max([label[1] for label in labels]) 
        visit = np.zeros(length)
        res = []
        for label in labels:
            if sum(visit[label[0]:label[1]]):
                label[3] = 1
            else:
               visit[label[0]:label[1]] = 1
            res.append(label)
        return res 
    else:
        return labels

## Preprocess function adapted from the Propaganda Detection paper
def read_data(directory):
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt','')
        ids.append(id)
        texts.append(f.read_text())
        labels.append(parse_label(f.as_posix().replace('.txt', '.labels.tsv')))
    # labels can be empty 
    return ids, texts, labels
