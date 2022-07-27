preprocess:
	python3 preprocess.py ./data/hatespeech_text_label_vote.csv ./data/train.csv

create-trial:
	mkdir trial_configurations/Experiment$(id)
	touch trial_configurations/Experiment$(id)/train.json
	mkdir trial_configurations/Experiment$(id)/DKNN
	touch trial_configurations/Experiment$(id)/DKNN/trial1.json

clean:
	rm -f *.e*
	rm -f *.o*
