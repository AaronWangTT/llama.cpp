"""Build the HIP mask regression fixture and run its host-only layout checks.
Usage: python3 -I build-inkling-hip-rawmask-fixture.py SOURCE_TREE LIBDIR ROCM_ROOT NEW_OUTPUT_DIR
The fixture.cpp returned by this script is its own executable source; GPU execution is separate.
"""
import os,shutil,subprocess,sys
from pathlib import Path

def span(file,begin,end):
    s=file.read_text()
    if s.count(begin)!=1 or s.count(end)!=1 or s.index(begin)>=s.index(end):
        raise ValueError('unknown predicate boundaries in '+str(file))
    return s[s.index(begin):s.index(end)]

def main():
    if len(sys.argv)!=5: raise SystemExit(__doc__)
    tree,lib,ro,out=map(lambda p:Path(p).resolve(),sys.argv[1:]); gg=tree/'ggml';cu=gg/'src/ggml-cuda';mq=cu/'inkling-maskepi-hip.cuh'
    # Reuse exactly the compiled byte-range and graph-layout tests; do not maintain a second matcher.
    s=mq.read_text();begin='static bool inkling_maskepi_separate';separate=s[s.index(begin):]
    if s.count(begin)!=1 or separate.count('static ')!=1: raise ValueError('unexpected byte-range helper')
    inc=('static bool inkling_rawmask_enabled(){return true;}\n'
         +span(cu/'inkling-gather-hip.cuh','static bool inkling_gather_layout','static bool inkling_gather_match')
         +span(mq,'static bool inkling_maskepi_match',begin)+separate
         +span(cu/'ggml-cuda.cu','static bool inkling_gather_disjoint','static bool inkling_gather_ready')
         +span(cu/'inkling-rawmask.cuh','static bool inkling_rawmask_layout','// True if the mask of the site i'))
    out.mkdir(exist_ok=False); (out/'rawmask-layout.inc').write_text(inc)
    shutil.copyfile(tree/'tests/inkling-hip-rawmask.cpp',out/'fixture.cpp')
    subprocess.run(['c++','-std=c++17','-O2','-I'+str(gg/'include'),'-I'+str(gg/'src'),'-I'+str(ro/'include'),str(out/'fixture.cpp'),'-L'+str(lib),'-L'+str(ro/'lib'),'-Wl,-rpath-link,'+str(lib),'-Wl,-rpath-link,'+str(ro/'lib'),'-lggml','-lggml-base','-lrocprofiler-sdk-roctx','-o',str(out/'fixture')],check=True)
    subprocess.run([str(out/'fixture'),'--layout'],env={**os.environ,'LD_LIBRARY_PATH':str(lib)+':'+str(ro/'lib')},check=True)
if __name__=='__main__':main()
