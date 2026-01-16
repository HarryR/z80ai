#!/bin/sh
# Build ZX Spectrum 48K TAP file for tinychat
zcat training-data.txt.gz | ../../feedme.py
../../buildz80tap.py -o CHAT.TAP
echo ""
echo "ZX Spectrum TAP file created: CHAT.TAP"
echo "Load in emulator or transfer to real hardware"
echo "In ZX Spectrum BASIC:"
echo "  LOAD \"\" CODE"
echo "  RANDOMIZE USR 32768"
