# Path: Makefile

test:
	pytest training/tests/

quality_checks:
	isort .
	black .
	pylint . --recursive=y --fail-under=9

train:
	cd training && python training_flow.py

deploy:
	docker-compose up -d

stop:
	docker-compose down

logs:
	docker-compose logs -f

setup:
	conda create -n credit-card-default-prediction python=3.8
	pip install -r training/requirements.txt
	pip install -r app/requirements.txt
	pip install pytest pylint black isort
