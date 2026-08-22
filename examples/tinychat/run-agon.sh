#!/bin/sh
# Build an Agon Light / eZ80 binary for tinychat
zcat training-data.txt.gz | ../../feedme.py
../../buildez80.py -o CHAT.bin
echo ""
echo "Agon eZ80 binary created: CHAT.bin"
echo "Copy it onto the SD card and run it by name at the MOS prompt:"
echo "  CHAT"
