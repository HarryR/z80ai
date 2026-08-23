#!/bin/sh
# Build an Agon Light / eZ80 binary for the guess game
set -eu
zcat training-data.txt.gz | ../../feedme.py
# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz
../../buildez80.py -m model.npz -o GUESS.bin
echo ""
echo "Agon eZ80 binary created: GUESS.bin"
echo "Copy it onto the SD card and run it by name at the MOS prompt:"
echo "  GUESS"
