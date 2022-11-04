start:
	cd /work/06782/ysu707/ls6/DkNN
	source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh

dev-maverick:
	idev -m 120 -p gtx
	conda activate DkNN

dev:
	idev -m 120 -p gpu-a100
	conda activate DkNN

taccinfo:
	/usr/local/etc/taccinfo

create-trial:
	mkdir trial_configurations/$(name)
	touch trial_configurations/$(name)/train.json
	touch trial_configurations/$(name)/eval_and_test.json
	mkdir trial_configurations/$(name)/DKNN
	touch trial_configurations/$(name)/DKNN/KD-Conformal.json
	touch trial_configurations/$(name)/DKNN/KD-Normal.json
	touch trial_configurations/$(name)/DKNN/LSH-Conformal.json
	touch trial_configurations/$(name)/DKNN/LSH-Conformal.json

clean:
	rm -f *.e*
	rm -f *.o*
	conda clean --all

compile_results:
	python3 compile_results.py ./result_locations.json compiled_results.csv
