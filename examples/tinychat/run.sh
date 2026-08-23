#!/bin/sh
set -eu
zcat training-data.txt.gz | ../../feedme.py
# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz
../../build.py -m model.npz -o CHAT.COM
../../cpm CHAT.COM