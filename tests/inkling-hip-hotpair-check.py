"""Compare HOTPAIR_PASS=check outputs from independent processes, one mode/route per call.
Usage: python3 -I check.py D82_LAYOUT.f32 ORIGINAL.f32 [HOTPAIR.f32 ...]
D82_LAYOUT must be generated with the pinned parent library, not the private build set to off.
"""
from pathlib import Path
import sys

CASES = ('route', 'no_pair', 'single_hot', 'offset_tail127')
TOKENS, SLOTS, ROWS = 256, 8, 2304
SIZE = len(CASES) * TOKENS * SLOTS * ROWS * 4


def compare(parent, other):
    if parent.samefile(other):
        raise ValueError(f'{other}: same file as parent')
    for path in (parent, other):
        if path.stat().st_size != SIZE:
            raise ValueError(f'{path}: expected {SIZE} bytes')
    with parent.open('rb') as a, other.open('rb') as b:
        for offset in range(0, SIZE, 1 << 20):
            size = min(1 << 20, SIZE - offset)
            x, y = a.read(size), b.read(size)
            if len(x) != size or len(y) != size:
                raise ValueError('short read')
            if x != y:
                i = next(i for i in range(size) if x[i] != y[i])
                word, byte = divmod(offset+i, 4)
                col, row = divmod(word, ROWS)
                token, slot = divmod(col, SLOTS)
                case, token = divmod(token, TOKENS)
                raise ValueError(f'{other}: {CASES[case]} token={token} slot={slot} row={row} '
                                 f'byte={byte}: parent {x[i]:02x} != result {y[i]:02x}')
        if a.read(1) or b.read(1):
            raise ValueError('trailing bytes')
    print(f'{other}: identical to {parent}, {SIZE} bytes')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    try:
        for result in sys.argv[2:]:
            compare(Path(sys.argv[1]), Path(result))
    except (OSError, ValueError) as e:
        sys.exit(str(e))
