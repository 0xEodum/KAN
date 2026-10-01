from pathlib import Path
import csv, hashlib, json, struct, sys, subprocess
import numpy as np

root=Path.cwd();e=root/'docs/evidence/m4'
cpu=json.loads((e/'cpu-performance-manifest.json').read_text())
for p,expected in cpu['sources_sha256'].items():
    assert hashlib.sha256((root/p).read_bytes()).hexdigest()==expected,p
snapshots=[]
for run in cpu['runs']:
    p=e/(run['name']+'.csv');assert hashlib.sha256(p.read_bytes()).hexdigest()==run['csv_sha256'],p
    rows=list(csv.DictReader(p.read_text().splitlines()));samples=[float(r['full_call_ms']) for r in rows if int(r['measured'])]
    assert samples==run['samples_ms'];assert [int(r['allocations']) for r in rows]==run['allocation_counts']
    p=root/'build-m4-cpu'/(run['name']+'.bin');data=p.read_bytes();assert len(data)==run['snapshot_bytes'];assert hashlib.sha256(data).hexdigest()==run['snapshot_sha256']
    offset=0;spans=[];count=0
    while offset<len(data):
      n=struct.unpack_from('<Q',data,offset)[0];offset+=8;v=np.frombuffer(data,dtype='<f8',count=n,offset=offset);assert np.isfinite(v).all();offset+=n*8;spans.append(n);count+=n
    assert spans==cpu['verification']['span_sizes'];assert count==392416;snapshots.append(data)
assert all(v==snapshots[0] for v in snapshots)
for name,expected in cpu['executables_sha256'].items():assert hashlib.sha256((root/'build-m4-cpu'/name).read_bytes()).hexdigest()==expected
for key in ['baseline','final']:
    samples=[v for r in cpu['runs'] if ('baseline' in r['name'])==(key=='baseline') for v in r['samples_ms']]
    assert np.median(samples)==cpu['summary'][key]['median_ms'];assert np.isclose(np.diff(np.quantile(samples,[.25,.75]))[0],cpu['summary'][key]['iqr_ms'])
print('PASS CPU provenance: all source/CSV/executable/snapshot hashes,392416 finite doubles bit-identical, six-sample pooled medians/IQRs')

gpu=json.loads((e/'balanced-final-summary.json').read_text())
for mode,stats in gpu['modes'].items():
    for source in ['preopt','final']:
      samples=[]
      for i in [1,2]:
        rows=list(csv.DictReader((e/f'balanced-{source}-{i}.csv').read_text().splitlines()));row=next(r for r in rows if r['backend']==mode)
        assert row['case']=='11' and row['batch']=='1024' and row['degrees']=='6/4';assert int(row['workspace_allocations'])==2;assert float(row['max_abs_error'])<=2e-10
        samples.extend(map(float,row['samples_ms'].split(';')))
      assert len(samples)==14 and samples==stats[source]['samples'];assert np.median(samples)==stats[source]['median_ms'];assert np.isclose(np.diff(np.quantile(samples,[.25,.75]))[0],stats[source]['iqr_ms'])
print('PASS GPU provenance: raw balanced source/fixture/mode rows,14 samples per source/mode, pooled medians/IQRs, VJP tolerance, allocations2')
for name,rows in [('final-m2-regression.csv',76),('final-m3-regression.csv',36),('final-m4.csv',36)]:
    records=list(csv.DictReader((e/name).read_text().splitlines()));assert len(records)==rows
    assert all(np.isfinite(float(r['max_abs_error'])) and float(r['max_abs_error'])<=2e-10 for r in records)
    assert all(int(r['workspace_allocations'])==2 for r in records if 'resident' in r['backend'] or 'transfer' in r['backend'])
print('PASS M2/M3/M4 full-call numerical sweeps:76/36/36 rows, tolerances and resident allocation counts')

manifest=e/'manifest.json'
if manifest.exists() and '--skip-manifest' not in sys.argv:
    m=json.loads(manifest.read_text())
    for group in [m['source']['files'],m['retained'],m['local_reproducible_not_committed']]:
      for item in group:
        p=root/item['path'];data=p.read_bytes();assert len(data)==item['bytes'],p;assert hashlib.sha256(data).hexdigest()==item['sha256'],p
    for item in m['source'].get('balanced_source_blobs',[]):
        spec=item['revision']+':'+item['path'];data=subprocess.check_output(['git','show',spec],cwd=root)
        assert len(data)==item['bytes'];assert hashlib.sha256(data).hexdigest()==item['sha256']
        assert subprocess.check_output(['git','rev-parse',spec],cwd=root,text=True).strip()==item['git_blob']
    print('PASS final M4 manifest: every recorded source, retained evidence and local executable/trace hash/size')
