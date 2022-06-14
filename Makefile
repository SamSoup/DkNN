conda-requirements:
	conda list -e > requirements.txt

pip3-requirements:
	pip3 freeze > requirements.txt
	
