#!/bin/sh
# Retrain the guess model from the checked-in data and run it under CP/M.
#
# To regenerate the data instead of using the shipped set (needs Ollama or an
# ANTHROPIC_API_KEY, and is not reproducible - see data/adventure/generate.py
# for one that is):
#   ./gendata.py -t 'chair' -d 30 -n 1000 --nonsense | tee -a chair.txt
#   cat chair.txt | ./balance.py -t 5000 -o --stats | gzip > training-data.txt.gz
set -eu

zcat training-data.txt.gz | ../../feedme.py

# Export to .npz: build-examples.sh, verify_artifacts.py, bench.py and the
# release all read model.npz, so skipping this leaves them on the old weights
# and the binary you just tested is not the one that ships.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz

../../build.py -m model.npz -o GUESS.COM
../../cpm GUESS.COM
