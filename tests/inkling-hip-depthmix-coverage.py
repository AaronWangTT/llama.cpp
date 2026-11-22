"""Fail-closed device coverage for a single, synchronized ROCm process. No HIP calls."""
from collections import Counter
import csv, re, sys
from pathlib import Path

def check(trace, mode):
    def rows(kind):
        f,=Path(trace).glob('*'+kind+'_trace.csv')
        with f.open(newline='') as s:return list(csv.DictReader(s))
    def times(r):return int(r['Start_Timestamp']),int(r['End_Timestamp'])
    kernels=[]
    for r in rows('kernel'):
        name=r['Kernel_Name']; kernel='inkling_depthmix_raw_v2' if mode in ('raw-model','raw-fixture') else 'inkling_depthmix_cont'
        if 'inkling_depthmix_' not in name:continue
        m=re.search(r'\b'+kernel+r'<\s*([24]),\s*([134])\s*>',name)
        dims=tuple(int(r[k]) for k in ('Grid_Size_X','Grid_Size_Y','Grid_Size_Z','Workgroup_Size_X','Workgroup_Size_Y','Workgroup_Size_Z'))
        if not m or dims!=(2304,256,1,128,1,1):raise ValueError((name,dims))
        kernels.append((times(r),tuple(map(int,m.groups()))))
    marks=rows('marker_api')
    if mode=='fixture':
        want={f'case_{s}_{q}_{k}_{j}':Counter({(s,q):1}) if j==0 else Counter() for s in (2,4) for q in (1,3,4) for k in range(2) for j in range(5)}
    elif mode=='raw-fixture':
        want={f'raw_{s}_{q}_{k}':Counter({(s,q):1}) for s in (2,4) for q in (1,3,4) for k in range(4)}
    elif mode in ('model','raw-model'):
        # block_size=6, core [6,18) x3, 24 physical layers. Previous-pass bank has S=2.
        per=Counter({(2,1):13,(2,3):1,(2,4):24,(4,1):31,(4,3):3})
        want={f'pp{i}':Counter({k:8*v for k,v in per.items()}) for i in range(2)}
    else:raise ValueError(mode)
    total=0
    for name,c in want.items():
        mark,=[v for v in marks if v['Function']==name]; a,b=times(mark)
        actual=Counter(shape for (start,end),shape in kernels if a<=start<=end<=b)
        if actual!=c:raise ValueError((name,'expected',dict(c),'device',dict(actual)))
        total+=sum(actual.values())
    if total!=len(kernels):raise ValueError(('dispatch outside allowed range',len(kernels),total))
    print('PASS',mode,'exact device coverage',total)
if __name__=='__main__':check(*sys.argv[1:])
