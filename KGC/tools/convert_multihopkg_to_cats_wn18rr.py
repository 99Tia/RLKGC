import json, os, tqdm

INPUT = "/home/ib5539/code/MultiHopKG/outputs/wn18rr_cats.jsonl"
OUTPUT = "/home/ib5539/code/CATS/datasets/WN18RR-subset-inductive/paths/close_path.json"
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

out_dict = {}
with open(INPUT) as fin:
    for ln in fin:
        obj = json.loads(ln)
        h, t = str(obj["query"]["e1_id"]), str(obj["candidate"]["e2_id"])
        key = f"{h}-{t}"
        all_paths = []
        for p in obj.get("paths", []):
            steps = p.get("steps", [])
            triples = []
            for s in steps:
                triples.append([str(s["e_id"]), str(s["r_id"]), str(s.get("next_e_id", s["e_id"]))])
            all_paths.append(triples)
        if all_paths:
            out_dict[key] = all_paths

with open(OUTPUT, "w", encoding="utf-8") as fout:
    json.dump(out_dict, fout, indent=2)
print(f"✅ Wrote {len(out_dict)} head-tail pairs -> {OUTPUT}")
