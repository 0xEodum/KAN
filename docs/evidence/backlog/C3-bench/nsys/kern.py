import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
steps=float(sys.argv[2]) if len(sys.argv)>2 else 1
tot=sum(float(r['Total Time (ns)']) for r in rows)
print(f"total GPU {tot/1e6/steps:.3f} ms/step")
for r in rows[:16]: print(f"{float(r['Time (%)']):5.1f}% {int(r['Instances'])/steps:5.1f}/step {float(r['Avg (ns)'])/1e3:9.1f}us  {r['Name'][:100]}")
