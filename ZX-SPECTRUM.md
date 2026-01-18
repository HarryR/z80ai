# ZX Spectrum 48K Support

This document describes how to build and run Z80-μLM on a ZX Spectrum 48K.

## Overview

The ZX Spectrum 48K port adapts the CP/M version to use ZX Spectrum ROM routines and memory layout. The resulting TAP files can be loaded in emulators or transferred to real hardware.

## Key Differences from CP/M Version

### Memory Layout
- **Origin Address**: 0x8000 (32768) instead of 0x0100
- Uses high memory to avoid overwriting BASIC system variables
- Compatible with ZX Spectrum 48K memory map

### I/O Routines
- **Character Output**: RST 0x10 (ROM print routine) instead of BDOS
- **Keyboard Input**: ROM routine at 0x10A8 (KEY_INPUT)
- **Screen Management**: ROM CLS at 0x0DAF

### File Format
- Generates `.TAP` files instead of `.COM` files
- TAP format includes header and data blocks with checksums
- Compatible with most ZX Spectrum emulators and tape interfaces

## Building for ZX Spectrum

### Prerequisites
- Python 3.6+
- NumPy
- PyTorch (only needed for `.pt` files; not required for `.npz`)
- Trained model file (`.pt` or `.npz`)

### Build Script

Use `buildz80tap.py` to build TAP files:

```bash
./buildz80tap.py --model command_model_autoreg.pt --output CHAT.TAP
```

Options:
- `-m, --model`: Path to trained model file (default: `command_model_autoreg.pt`)
- `-o, --output`: Output TAP filename (default: `CHAT.TAP`)

### Example: Building tinychat

```bash
cd examples/tinychat
./run-zx.sh
```

This will:
1. Decompress training data
2. Train the model with `feedme.py`
3. Build the TAP file with `buildz80tap.py`
4. Output: `CHAT.TAP`

### Example: Building guess game

```bash
cd examples/guess
./run-zx.sh
```

Outputs: `GUESS.TAP`

## Loading and Running

### In an Emulator

Most ZX Spectrum emulators support TAP files:

1. **FUSE** (Free Unix Spectrum Emulator):
   ```bash
   fuse --tape CHAT.TAP
   ```
   Then in BASIC:
   ```basic
   LOAD "" CODE
   RANDOMIZE USR 32768
   ```

2. **ZEsarUX**:
   - File → Load binary file → Select CHAT.TAP
   - Or use Smart Load feature

3. **Speccy** (Windows):
   - File → Open → Select CHAT.TAP
   - F3 to start tape

### On Real Hardware

Transfer TAP files to real ZX Spectrum using:

1. **TZXDuino**: Audio cassette interface
2. **DivMMC/DivIDE**: SD card interface
3. **ZX Interface 1**: Microdrive or tape interface
4. **Audio Cable**: Convert TAP to WAV and play through audio

### Loading Instructions

Once the TAP is loaded (via emulator or real hardware):

```basic
LOAD "" CODE
```

Wait for loading to complete, then run:

```basic
RANDOMIZE USR 32768
```

The program will:
1. Clear the screen
2. Display a `>` prompt
3. Wait for input

### Usage

**Interactive Chat Mode**:
```
> hello
HI
> are you a robot
YES
> do you dream
MAYBE
> !
```

Type `!` to exit back to BASIC.

## Technical Details

### Memory Usage

Typical memory layout for a 256→192→128→64 architecture:

| Section | Size | Address Range | Description |
|---------|------|---------------|-------------|
| Code | ~5 KB | 0x8000-0x93FF | Z80 machine code |
| Variables | ~100 bytes | 0x9400-0x9463 | Runtime variables |
| Input Buffer | 62 bytes | 0x9464-0x94A1 | User input |
| Token Buffer | 512 bytes | 0x94A2-0x96A1 | Trigram buckets (256 × 2) |
| Hidden Buffers | ~800 bytes | 0x96A2-0x99C1 | Layer activations |
| Output Buffer | 128 bytes | 0x99C2-0x9A41 | Character scores |
| Weights | ~28 KB | 0x9A42-0xFFFF | 2-bit quantized weights |

**Total**: ~35-40 KB (fits comfortably in 48K)

### Performance

On a 3.5 MHz Z80:
- **Inference time**: ~1-2 seconds per character
- **Input processing**: Near-instant
- **Total response time**: 2-10 seconds for typical outputs

### Compatibility

**Tested on**:
- ZX Spectrum 48K (original and +)
- FUSE emulator
- ZEsarUX emulator

**Should work on**:
- ZX Spectrum 128K (48K mode)
- ZX Spectrum +2/+2A/+3 (48K mode)
- Pentagon 128

**Not compatible with**:
- ZX Spectrum 16K (insufficient memory)
- QL or other Sinclair systems (different architecture)

## Limitations

### Compared to CP/M Version

1. **No command-line arguments**: Always starts in interactive mode
2. **Keyboard handling**: Uses ZX Spectrum keyboard matrix
   - May differ slightly from CP/M keyboard behavior
   - ENTER key handling is native to ZX Spectrum
3. **Character set**: Limited to ZX Spectrum printable characters
4. **No file I/O**: Models must be embedded at compile time

### General Limitations

Same as CP/M version:
- Maximum 50 characters per response
- Trigram encoding limitations (word order insensitive)
- 2-bit quantized weights (limited expressiveness)
- Small model capacity

## Optimization Tips

### Model Size

To fit larger models or reduce memory usage:

1. **Reduce hidden layer sizes**:
   ```python
   # In feedme.py or training script
   hidden_sizes = [128, 96]  # Instead of [192, 128]
   ```

2. **Simplify architecture**:
   ```python
   hidden_sizes = [128]  # Single hidden layer
   ```

3. **Reduce charset**:
   - Remove uncommon characters
   - Keep only uppercase + space + punctuation

### Speed

The standard `buildz80tap.py` uses packed 2-bit weights (slower but smaller).

For ~10x faster inference with larger file size:
- Port `buildfastz80com.py` optimizations
- Uses skip lists for non-zero weights
- Trades ~5KB extra size for significant speed gain
- Recommended for ZX Spectrum 128K

## Troubleshooting

### "Out of Memory" during build

- Reduce model size (fewer/smaller hidden layers)
- Reduce charset size
- Use Python with more available RAM

### TAP file won't load

- Verify TAP file integrity
- Try different emulator
- Check file wasn't corrupted during transfer

### Program crashes on run

- Ensure using `RANDOMIZE USR 32768`
- Check model was built correctly
- Verify sufficient memory (48K required)

### Garbled output

- Character set mismatch
- ROM routine compatibility issue
- Try rebuilding TAP file

### No input response

- Check keyboard input routine
- Emulator keyboard mapping may differ
- Try real hardware if using emulator

## Building from Scratch

Complete workflow from training to TAP:

```bash
# 1. Generate training data (example: tinychat)
cd examples/tinychat
python3 genpairs.py > training-data.txt

# 2. Train model
cat training-data.txt | ../../feedme.py

# 3. Build ZX Spectrum TAP
../../buildz80tap.py -m command_model_autoreg.pt -o CHAT.TAP

# 4. Test in emulator
fuse --tape CHAT.TAP
```

## Further Reading

- [ZX Spectrum TAP Format](https://sinclair.wiki.zxnet.co.uk/wiki/TAP_format)
- [ZX Spectrum ROM Routines](https://skoolkid.github.io/rom/)
- [Z80 Programming](http://www.z80.info/)
- Main README: [README.md](README.md)
- Training guide: [TRAINING.md](TRAINING.md)

## License

Same as main project: MIT or Apache-2.0
