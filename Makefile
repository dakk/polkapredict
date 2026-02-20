VENV := $(HOME)/venv
PYTHON := $(VENV)/bin/python

.PHONY: all polkadot kusama selfstake election selfstake-polkadot selfstake-kusama election-polkadot election-kusama serve clean

all: polkadot

polkadot: selfstake-polkadot election-polkadot

kusama: election-kusama

selfstake: selfstake-polkadot

election: election-polkadot

selfstake-polkadot:
	$(PYTHON) polkadot_zero_selfstake.py --chain polkadot

selfstake-kusama:
	$(PYTHON) polkadot_zero_selfstake.py --chain kusama

election-polkadot:
	$(PYTHON) polkadot_election_prediction.py --chain polkadot

election-kusama:
	$(PYTHON) polkadot_election_prediction.py --chain kusama

serve:
	$(PYTHON) -m http.server 8000

clean:
	rm -rf data/
