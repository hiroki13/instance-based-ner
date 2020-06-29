# Instance-Based Named Entity Recognizer

This codebase is partially based on [neural_sequence_labeling](https://github.com/IsaacChanghau/neural_sequence_labeling)

## Citation
* Instance-Based Learning of Span Representations: A Case Study through Named Entity Recognition
* Hiroki Ouchi, Jun Suzuki, Sosuke Kobayashi, Sho Yokoi, Tatsuki Kuribayashi, Ryuto Konno, Kentaro Inui
* In ACL 2020
* Conference paper: https://www.aclweb.org/anthology/2020.acl-main.575/
* arXiv version: https://arxiv.org/abs/2004.14514

@inproceedings{ouchi-etal-2020-instance,
    title = "Instance-Based Learning of Span Representations: A Case Study through Named Entity Recognition",
    author = "Ouchi, Hiroki  and
      Suzuki, Jun  and
      Kobayashi, Sosuke  and
      Yokoi, Sho  and
      Kuribayashi, Tatsuki  and
      Konno, Ryuto  and
      Inui, Kentaro",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.575",
    pages = "6452--6459",
    abstract = "Interpretable rationales for model predictions play a critical role in practical applications. In this study, we develop models possessing interpretable inference process for structured prediction. Specifically, we present a method of instance-based learning that learns similarities between spans. At inference time, each span is assigned a class label based on its similar spans in the training set, where it is easy to understand how much each training instance contributes to the predictions. Through empirical analysis on named entity recognition, we demonstrate that our method enables to build models that have high interpretability without sacrificing performance.",
}

## Prerequisites
* [python3](https://www.python.org/downloads/)
* [TensorFlow](https://www.tensorflow.org/)
* [h5py](https://www.h5py.org/)

## Installation
- CPU
```
conda create -n instance-based-ner python=3.6
source activate instance-based-ner
conda install -c conda-forge tensorflow
pip install ujson tqdm
git clone https://github.com/cl-tohoku/instance-based-ner_dev.git
```
- GPU
```
conda create -n instance-based-ner python=3.6
source activate instance-based-ner
pip install tensorflow-gpu==1.10 ujson tqdm
git clone https://github.com/cl-tohoku/instance-based-ner_dev.git
```

## Data Preparation
`./create_datasets.sh`

## Pretrained Models
* [Instance-based span model](https://drive.google.com/open?id=1d_KzED0UKEVnorymxiylzEOHpXoFF8TN)
* [Classifier-based span model](https://drive.google.com/open?id=16MFR1IQ5mPx0bFAXMxdEbnTEn8zO2RNX)

## Get Started
`python run_knn_models.py --mode cmd --config_file checkpoint_knn_conll2003_lstm-minus_batch8_keep07_0/config.json`

## Usage
### Instance-based span model
* Training: `python train_knn_models.py --config_file data/config/config.knn.conll2003.json`
* Predicting with random training sentences: `python run_knn_models.py --config_file checkpoint_knn/conll2003/config.json --knn_sampling random --data_path data/conll2003/valid.json`
* Predicting with nearest training sentences: `python run_knn_models.py --config_file checkpoint_knn/conll2003/config.json --knn_sampling random --data_path data/conll2003/valid.glove.50-nn.json`
### Classifier-based span model
* Training: `python train_span_models.py --config_file data/config/config.span.conll2003.json`
* Predicting: `python run_span_models.py --config_file checkpoint_span/conll2003/config.json --data_path data/conll2003/valid.json`

## LICENSE
MIT License
