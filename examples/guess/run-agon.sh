#!/bin/sh
# Build an Agon Light / eZ80 binary for the guess game
zcat training-data.txt.gz | ../../feedme.py
../../buildez80.py -o GUESS.bin
echo ""
echo "Agon eZ80 binary created: GUESS.bin"
echo "Copy it onto the SD card and run it by name at the MOS prompt:"
echo "  GUESS"
