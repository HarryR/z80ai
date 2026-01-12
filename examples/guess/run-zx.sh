#!/bin/sh
# Build ZX Spectrum 48K TAP file for guess game
zcat training-data.txt.gz | ../../feedme.py
../../buildz80tap.py -o GUESS.TAP
echo ""
echo "ZX Spectrum TAP file created: GUESS.TAP"
echo "Load in emulator or transfer to real hardware"
echo "In ZX Spectrum BASIC:"
echo "  LOAD \"\" CODE"
echo "  RANDOMIZE USR 32768"
