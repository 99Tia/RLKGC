import json

root = "datasets/UMLS-inductive"
with open(f"{root}/entity2text.txt") as f:
    e2txt = dict(line.strip().split('\t',1) for line in f if line.strip())
with open(f"{root}/relation2text.txt") as f:
    r2txt = dict(line.strip().split('\t',1) for line in f if line.strip())
with open(f"{root}/paths/close_path.json") as f:
    cp = json.load(f)

# pick one head-tail
k = next(iter(cp.keys()))
print("key:", k)
for path in cp[k][:2]:   # show first 2 paths
    pretty = " -> ".join(f"({e2txt[h]}, {r2txt[r]}, {e2txt[t]})" for h,r,t in path)
    print(pretty)
