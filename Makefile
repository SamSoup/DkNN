preprocess:
	python3 preprocess.py ./data/hatespeech_text_label_vote.csv ./data/train.csv

run:
	python3 main.py \
	--model_name_or_path distilbert-base-uncased \
	--train_file ./data/train.csv \
	--seed 42 \
	--shuffle_seed 42 \
	--output_dir ./output/distilbert-base-uncased-2 \
	--do_train_val_test_split \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--do_predict \
	--pad_to_max_length \
	--input_key Tweet \
	--report_to none \
	--learning_rate 1e-5 \
	--num_train_epochs 10 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \

clean:
	rm -f *.e*
	rm -f *.o*
