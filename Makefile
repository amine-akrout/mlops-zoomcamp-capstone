# Path: Makefile

test:
	pytest training/tests/

quality_checks:
	isort .
	black .
	pylint .\training --recursive=y --fail-under=9

train:
	cd training && python training_flow.py

deploy:
	docker-compose up -d

stop:
	docker-compose down

logs:
	docker-compose logs -f
