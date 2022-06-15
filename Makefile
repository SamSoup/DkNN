preprocess:
	python3 preprocess.py ./data/hatespeech_text_label_vote.csv ./data/train.csv

run:
	python3 run_glue.py \
	--train_file ./data/train.csv \
	--seed 42 \
	--data_seed 42\
	--do_train_val_test_split \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--do_predict \