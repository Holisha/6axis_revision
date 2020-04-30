all:
	nohup python main.py > model_solution.txt
	python axis2img.py
