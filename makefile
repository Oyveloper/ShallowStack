game:
	poetry run python3 main.py game

debug:
	poetry run python3 main.py --debug game

dataset:
	poetry run python3 main.py generate-training-data

test-stuff:
	poetry run python3 main.py test-stuff

cheat-sheet:
	poetry run python3 main.py generate-cheat-sheet

show-cheat-sheet:
	poetry run python3 main.py show-cheat-sheet

hand-types:
	poetry run python3 main.py gen-hand-types

test:
	poetry run pytest

train:
	poetry run python3 main.py  train-all --data_size 50 --override_river --epochs 20

tensorboard:
	tensorboard --logdir lightning_logs
