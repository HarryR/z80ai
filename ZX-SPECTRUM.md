# ZX Spectrum 48K Support

This document describes how to build and run Z80-μLM on a ZX Spectrum 48K.

## Overview

The ZX Spectrum 48K port adapts the CP/M version to use ZX Spectrum ROM routines and memory layout. The resulting TAP files can be loaded in emulators or transferred to real hardware.

## Key Differences from CP/M Version

### Memory Layout
- **Origin Address**: 0x6000 (24576) instead of 0x0100
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
   CLEAR 24575
   LOAD "" CODE
   RANDOMIZE USR 24576
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
CLEAR 24575
LOAD "" CODE
```

Wait for loading to complete, then run:

```basic
RANDOMIZE USR 24576
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

The image is one contiguous block loaded at `ORG_ADDR`, laid out as code,
then runtime variables and buffers, then the weights — which dominate.

| Section | Size | Description |
|---------|------|-------------|
| Code | ~2.5 KB | Z80 machine code |
| Variables + input buffer | ~90 bytes | Runtime state, 62-byte input line |
| Token buffer | 512 bytes | 256 buckets × 2 bytes (128 query + 128 context) |
| Hidden buffers | 2 × max layer × 2 bytes | Ping-ponged layer activations |
| Output buffer | charset × 2 bytes | Character scores |
| Weights + biases | remainder | 2-bit packed weights, 16-bit biases |

The load address bounds how large a model can be, since RAM ends at `0xFFFF`:

| Load address | Available |
|---|---|
| `0x6000` (default) | 40,960 bytes |
| `0x8000` (pre-fix) | 32,768 bytes |

The two shipped examples assemble to 38,949 and 40,022 bytes, so **neither fits
above `0x8000`** — that was a real bug, and `.TAP` files built before the move
to `0x6000` ran past the end of the address space and could not load at all.
`buildz80tap.py` now refuses to emit an image that would not fit, and reports
the headroom left:

```
Loads at 0x6000-0xf824, 2,011 bytes of RAM to spare
```

If you need more room, either lower `--org` (0x6000 is already just above the
system variables) or train a narrower model.

### Performance

Measured with `bench.py`, which runs the build in an emulator and counts cycles.
For the shipped 256→256→192→128→11 model on a 3.5 MHz Z80:

- **Inference**: 26,843,795 T-states per character — about **7.7 seconds**
- **Total response**: roughly 7.7 seconds × the number of characters emitted

The ZX build shares its inner loop with the CP/M one (`libnn.emit_layer`). It
used to carry its own slower copy, which cost 36% more instructions per
character and was where the `MULADD` borrow bug survived a first fix.

The CP/M `buildfastz80com.py` index-list layout is around 9x quicker again; it
has not been ported to the ZX build, where the 40,960-byte ceiling makes the
extra size hard to afford.

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

- Ensure using `RANDOMIZE USR 24576`
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
