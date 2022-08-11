preprocess:
	python3 preprocess.py ./data/hatespeech_text_label_vote.csv ./data/train.csv

create-trial:
	mkdir trial_configurations/$(name)
	touch trial_configurations/$(name)/train.json
	touch trial_configurations/$(name)/eval.json
	touch trial_configurations/$(name)/test.json
	mkdir trial_configurations/$(name)/DKNN
	touch trial_configurations/$(name)/DKNN/trial1.json

clean:
	rm -f *.e*
	rm -f *.o*
	conda clean --all
