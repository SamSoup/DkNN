preprocess:
	python3 preprocess.py ./data/hatespeech_text_label_vote.csv ./data/train.csv

run:
	python3 main.py \
	--model_name_or_path vinai/bertweet-large \
	--train_file ./data/train.csv \
	--seed 42 \
	--shuffle_seed 42 \
	--output_dir ./output/bert \
	--do_train_val_test_split \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--do_predict \
