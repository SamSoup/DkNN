start:
	cd /work/06782/ysu707/ls6/DkNN
	source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh

simplify-models-toxigen:
	python3 SimplifyModel.py ./result_locations_toxigen.json 16

setup-trials:
	python3 SetUpDKNNTrials.py ./trial_configurations/toxigen/deberta-large-1-new/base.json ./trial_configurations/toxigen/deberta-large-1-new/ 25
	python3 SetUpDKNNTrials.py ./trial_configurations/toxigen/bart-large-1-new/base.json ./trial_configurations/toxigen/bart-large-1-new/ 26

dev-maverick:
	idev -m 120 -p gtx
	conda activate DkNN

dev:
	idev -m 120 -p gpu-a100
	conda activate DkNN

taccinfo:
	/usr/local/etc/taccinfo

clean:
	rm -f *.e*
	rm -f *.o*
	conda clean --all

compile_results:
	python3 compile_results.py ./result_locations.json compiled_results.csv

compile_results_toxigen:
	python3 compile_results_new.py ./result_locations_toxigen.json compiled_results_toxigen.csv

evaluate_explanations_toxigen:
	python3 EvaluateExplanations.py ./result_locations_toxigen.json

start-up-in-maverick-notebook:
	ln -s /work/06782/ysu707/ls6/DkNN /work/06782/ysu707/maverick2/LS6WORK
	cd /work/06782/ysu707/ls6/DkNN
	source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
