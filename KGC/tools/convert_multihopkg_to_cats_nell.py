#!/usr/bin/env python3
import os, json

# INPUT: MultiHopKG jsonl
INPUT = "/home/ib5539/code/MultiHopKG/outputs/nell995_cats.jsonl"

# OUTPUT: CATS close_path.json (inductive)
OUTPUT = "/home/ib5539/code/CATS/datasets/NELL-995-subset-inductive/paths/close_path.json"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

def normalize_step(step):
    """
    MultiHopKG step fields can be either id/text.
    We only need IDs (strings). Fall back to text if id missing.
    """
    r = step.get("r_id", step.get("r"))
    e = step.get("e_id", step.get("e"))
    return str(r), str(e)

def main():
    count_pairs = 0
    count_paths = 0
    paths_dict = {}  # key: "head-tail" -> list of paths, each path is [[h,r,t], ...]
    with open(INPUT, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("query", {})
            cand = rec.get("candidate", {})
            e1 = str(q.get("e1_id", q.get("e1")))
            e2 = str(cand.get("e2_id", cand.get("e2")))
            key = f"{e1}-{e2}"

            all_paths = []
            for p in rec.get("paths", []):
                steps = p.get("steps", [])
                if not steps:
                    continue

                # Build triples along the path: e0 --r0--> e1 --r1--> e2 ...
                triples = []
                cur_head = e1
                for st in steps:
                    r_id, next_e = normalize_step(st)
                    triples.append([cur_head, r_id, next_e])
                    cur_head = next_e
                # Keep only paths that end at candidate e2
                if cur_head == e2 and len(triples) > 0:
                    all_paths.append(triples)

            if all_paths:
                paths_dict.setdefault(key, []).extend(all_paths)
                count_pairs += 1
                count_paths += len(all_paths)

    with open(OUTPUT, "w", encoding="utf-8") as fout:
        json.dump(paths_dict, fout)
    print(f"✅ Wrote {count_pairs} pairs, {count_paths} total paths -> {OUTPUT}")

if __name__ == "__main__":
    main()
