import json
from collections import defaultdict

INPUT  = "/home/ib5539/code/MultiHopKG/outputs/fb15k237_cats.jsonl"
OUTPUT = "/home/ib5539/code/CATS/datasets/FB15k-237-subset-inductive/paths/close_path.json"
STORE_REVERSE = True  # also store reversed paths (tail-head)

def main():
    close_path = defaultdict(list)
    n_pairs = n_paths = n_bad = 0

    with open(INPUT, "r", encoding="utf-8") as fin:
        for ln, line in enumerate(fin, 1):
            if not line.strip():
                continue
            obj = json.loads(line)

            e1 = obj.get("query", {}).get("e1_id")
            e2 = obj.get("candidate", {}).get("e2_id")
            paths = obj.get("paths", [])

            if e1 is None or e2 is None or not isinstance(paths, list):
                n_bad += 1
                continue

            key = f"{e1}-{e2}"
            rev = f"{e2}-{e1}"

            added_any = False
            for p in paths:
                steps = p.get("steps", [])
                if not isinstance(steps, list) or not steps:
                    continue

                cur = e1
                triples = []
                ok = True
                for st in steps:
                    # Use numeric IDs to match FB15k-237-subset files
                    r = st.get("r_id")
                    nxt = st.get("e_id")
                    if r is None or nxt is None:
                        ok = False
                        break
                    triples.append([str(cur), str(r), str(nxt)])
                    cur = nxt

                if not ok:
                    continue

                # optionally check that the path ends at candidate e2
                if cur != e2:
                    # keep anyway (some finders produce alternative tails)
                    pass

                close_path[key].append(triples)
                if STORE_REVERSE:
                    # reverse order & swap head/tail per hop
                    rev_triples = [[t, r, h] for (h, r, t) in triples[::-1]]
                    close_path[rev].append(rev_triples)

                n_paths += 1
                added_any = True

            if added_any:
                n_pairs += 1

    # write
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        json.dump(close_path, fout, indent=2)

    print(f"✅ Wrote {n_pairs} pairs, {n_paths} total paths -> {OUTPUT}")
    if n_bad:
        print(f"   Skipped {n_bad} malformed lines")

if __name__ == "__main__":
    main()
