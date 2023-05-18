# aics-social-networks-analysis
aics-social-networks-analysis project

Dataset from https://snap.stanford.edu/data/sx-stackoverflow.html

## Initialize (don't need to run)
```
poetry init
poetry config --local virtualenvs.in-project true
poetry add pandas
poetry install
```

## Run first time
```
poetry install
poetry run python social_networks_analysis/main.py
```

## Run project
```
poetry run python social_networks_analysis/main.py
```
