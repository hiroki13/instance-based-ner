#! /bin/bash

# Download the GloVe embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir data/emb
mv glove.6B/* data/emb/

# Download the CoNLL-2003 dataset
git clone https://github.com/IsaacChanghau/neural_sequence_labeling.git
mkdir data/conll2003
mv neural_sequence_labeling/data/raw/conll2003/raw/* data/conll2003/
python scripts/convert_conll03_to_json.py --input_file data/conll2003/train.txt --output_file data/conll2003/train.json --remove_duplicates
python scripts/convert_conll03_to_json.py --input_file data/conll2003/valid.txt --output_file data/conll2003/valid.json
python scripts/convert_conll03_to_json.py --input_file data/conll2003/test.txt --output_file data/conll2003/test.json
# Valid/Test sets with nearest training sentences
python scripts/retrieve_knn_sents_with_glove.py --train_json data/conll2003/train.json --test_json data/conll2003/valid.json --glove data/emb/glove.6B.100d.txt --k 50 --output_file data/conll2003/valid.glove.50-nn.json
python scripts/retrieve_knn_sents_with_glove.py --train_json data/conll2003/train.json --test_json data/conll2003/test.json --glove data/emb/glove.6B.100d.txt --k 50 --output_file data/conll2003/test.glove.50-nn.json

# Download the GENIA dataset
git clone https://github.com/thecharm/boundary-aware-nested-ner.git
mkdir data/genia
mv boundary-aware-nested-ner/Our_boundary-aware_model/data/genia/* data/genia/
python scripts/convert_genia_to_json.py --input_file data/genia/genia.train.iob2 --output_file data/genia/train.json --remove_duplicates
python scripts/convert_genia_to_json.py --input_file data/genia/genia.dev.iob2 --output_file data/genia/valid.json
python scripts/convert_genia_to_json.py --input_file data/genia/genia.test.iob2 --output_file data/genia/test.json
# Valid/Test sets with nearest training sentences
python scripts/retrieve_knn_sents_with_glove.py --train_json data/genia/train.json --test_json data/genia/valid.json --glove data/emb/glove.6B.100d.txt --k 50 --output_file data/genia/valid.glove.50-nn.json
python scripts/retrieve_knn_sents_with_glove.py --train_json data/genia/train.json --test_json data/genia/test.json --glove data/emb/glove.6B.100d.txt --k 50 --output_file data/genia/test.glove.50-nn.json
