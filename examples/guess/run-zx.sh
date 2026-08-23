#!/bin/sh
# Build ZX Spectrum 48K TAP file for guess game
set -eu
zcat training-data.txt.gz | ../../feedme.py
# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz
../../buildz80tap.py -m model.npz -o GUESS.TAP
echo ""
echo "ZX Spectrum TAP file created: GUESS.TAP"
echo "Load in emulator or transfer to real hardware"
echo "In ZX Spectrum BASIC:"
echo "  CLEAR 24575"
echo "  LOAD \"\" CODE"
echo "  RANDOMIZE USR 24576"
