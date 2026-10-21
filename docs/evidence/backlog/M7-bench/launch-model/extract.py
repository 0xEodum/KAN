import sqlite3,sys,collections,statistics,json
db=sys.argv[1]; out=sys.argv[2]
c=sqlite3.connect(db)
q="""select coalesce(n.text, s2.value), s.value, k.end-k.start from CUPTI_ACTIVITY_KIND_KERNEL k
join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId=r.correlationId
join NVTX_EVENTS n on r.start between n.start and n.end and n.eventType=59
left join StringIds s2 on n.textId=s2.id
join StringIds s on k.shortName=s.id"""
d=collections.defaultdict(lambda: collections.defaultdict(list))
for name,kern,dur in c.execute(q):
    d[name][kern].append(dur)
res={}
for name,ks in d.items():
    reps=len(ks['wavelet_forward_kernel'])
    def tot(names):
        return sum(sum(ks[n]) for n in names if n in ks)/reps/1000.0  # us per rep
    res[name]={'fwd':tot(['wavelet_forward_kernel','wavelet_forward_finish_kernel']),
               'fwd_kernel':statistics.median(ks['wavelet_forward_kernel'])/1000,
               'fwd_finish':tot(['wavelet_forward_finish_kernel']),
               'bwd':tot(['wavelet_backward_kernel','wavelet_input_finish_kernel']),
               'bwd_finish':tot(['wavelet_input_finish_kernel']),
               'nfwd': (1 if 'wavelet_forward_finish_kernel' in ks else 0), 'nbwd': (1 if 'wavelet_input_finish_kernel' in ks else 0)}
json.dump(res,open(out,'w'))
print(len(res))
