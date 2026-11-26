"""Check synchronized d82 PP traces, or two isolated single-payload HIP fixtures."""
from collections import Counter
import csv
from pathlib import Path
import re
import runpy
import sys

PACK = re.compile(r"\binkling_concat4_pack\b")
CONCAT = re.compile(r"\bconcat_cont<\s*(?:unsigned int|uint32_t)\s*,\s*1\s*>")
DIMS = ("Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z", "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z")
PACK_DIMS = (2304, 4, 256, 256, 1, 1)


def read(trace, kind):
    path, = Path(trace).glob(f"*{kind}_trace.csv")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def kernels(trace):
    for row in read(trace, "kernel"):
        name = row["Kernel_Name"]
        pack = bool(PACK.search(name))
        dims = tuple(int(row[k]) for k in DIMS) if pack or CONCAT.search(name) else None
        if "inkling_concat4" in name and (not pack or dims != PACK_DIMS):
            raise ValueError((name, dims))
        yield int(row["Start_Timestamp"]), int(row["End_Timestamp"]), pack, dims


def model(trace, on):
    # Reuse the unmodified d82 gate: exact S/Q counts, grids and range attribution.
    runpy.run_path(str(Path(__file__).with_name("inkling-hip-depthmix-coverage.py")))["check"](trace, "raw-model")
    ks = list(kernels(trace))
    marks = read(trace, "marker_api")
    answer = []
    last_end = None
    for pp in ("pp0", "pp1"):
        mark, = (r for r in marks if r["Function"] == pp)
        a, b = int(mark["Start_Timestamp"]), int(mark["End_Timestamp"])
        if a >= b or (last_end is not None and a <= last_end):
            raise ValueError((pp, "invalid/overlapping PP ranges"))
        last_end = b
        inside = [(pack, dims) for start, end, pack, dims in ks if a <= start <= end <= b]
        n = sum(pack for pack, _ in inside)
        if n != (272 if on else 0):
            raise ValueError((pp, "pack", n))
        answer.append((len(inside), Counter(d for pack, d in inside if not pack and d is not None)))
    if sum(pack for _, _, pack, _ in ks) != (544 if on else 0):
        raise ValueError("pack outside expected ranges")
    return answer


def main(first, second, third=None):
    if first == "--fixture":
        if third is None or Path(second).samefile(third):
            raise ValueError("need separate trace directories for payloads 0 and 1")
        for trace in (second, third):
            if sum(pack for _, _, pack, _ in kernels(trace)) != 1:
                raise ValueError((trace, "expected one HIP pack"))
        print("PASS one pack in each single-payload HIP trace")
        return
    if third is not None:
        raise ValueError("expected OFF_TRACE ON_TRACE")
    controls, candidates = model(first, False), model(second, True)
    for (ref_total, ref), (alt_total, alt) in zip(controls, candidates):
        delta = ref.copy(); delta.subtract(alt)
        got = {d: n for d, n in delta.items() if n}
        want = {(2304*s*256, 1, 1, 256, 1, 1): 272 for s in (2, 3, 4)}
        if got != want or ref_total - alt_total != 544:
            raise ValueError(("CONCAT/total dispatch deltas", got, want, ref_total - alt_total))
    print("PASS 544 packs, 1088 net launches removed in PP ranges, all 1152 original weighted sums retained")


if __name__ == "__main__":
    main(*sys.argv[1:])
