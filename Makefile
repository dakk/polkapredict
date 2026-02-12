VENV := $(HOME)/venv
PYTHON := $(VENV)/bin/python

.PHONY: all selfstake election serve clean

all: selfstake election

selfstake:
	$(PYTHON) polkadot_zero_selfstake.py

election:
	$(PYTHON) polkadot_election_prediction.py

serve:
	$(PYTHON) -m http.server 8000

clean:
	rm -rf data/
