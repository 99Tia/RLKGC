import argparse, os, json
from collections import defaultdict

def read_id_map(path):
    """Reads a 2-col TSV: name\tid OR id\tname; returns two dicts."""
    name2id, id2name = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            a,b = line.strip().split("\t")[:2]
            # detect which column is id
            if a.isdigit() and not b.isdigit():
                _id, _name = int(a), b
            elif b.isdigit() and not a.isdigit():
                _name, _id = a, int(b)
            else:
                # fallback: assume a=name, b=id
                try:
                    _id = int(b); _name = a
                except:
                    try:
                        _id = int(a); _name = b
                    except:
                        raise ValueError(f"Unrecognized id-map line: {line}")
            name2id[_name] = _id
            id2name[_id] = _name
    return name2id, id2name

def load_triples_any(path):
    """
    Reads triples in either text form (e h t) or id form (int int int), space/tsv separated.
    Returns list of (h, r, t) as **ids** (ints).
    """
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().replace("\t", " ").split()
            if len(parts) != 3:
                # some datasets are comma separated; try that
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Bad triple line in {path}: {line}")
            triples.append(tuple(parts))
    return triples

def map_to_ids(triples, e_name2id, r_name2id):
    """
    Map a list of string/int triples to id triples (h_id, r_id, t_id).
    Accepts either (h, r, t) or (h, t, r) and auto-detects which is which.
    """
    out = []
    for a, b, c in triples:
        # all are digits already?
        if str(a).isdigit() and str(b).isdigit() and str(c).isdigit():
            # we still need to guess order; try (h,r,t) first, else (h,t,r)
            h, r, t = int(a), int(b), int(c)
            if r in r_name2id.values():
                out.append((h, r, t))
            else:
                # interpret as (h, t, r)
                out.append((h, t, r))
            continue

        # string case: try (h,r,t)
        if (a in e_name2id) and (b in r_name2id) and (c in e_name2id):
            out.append((e_name2id[a], r_name2id[b], e_name2id[c]))
            continue

        # try (h,t,r)
        if (a in e_name2id) and (b in e_name2id) and (c in r_name2id):
            out.append((e_name2id[a], r_name2id[c], e_name2id[b]))
            continue

        # if still ambiguous, raise with a helpful hint
        raise KeyError(
            f"Cannot map triple {a, b, c}. "
            f"Expected (h,r,t) or (h,t,r) using names in maps "
            f"(entities: {len(e_name2id)}, relations: {len(r_name2id)})."
        )
    return out

def write_triples(path, triples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h,r,t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhkg_root", required=True, help="Path to MultiHopKG repo (with data/umls)")
    ap.add_argument("--cats_root", required=True, help="Path to CATS repo")
    ap.add_argument("--dataset_dir", default="datasets/UMLS-inductive", help="CATS dataset dir to write into")
    ap.add_argument("--use_dev_as_valid", action="store_true", help="If true, writes dev.triples to valid.txt")
    args = ap.parse_args()

    # ---- MultiHopKG input files (UMLS) ----
    data_dir = os.path.join(args.mhkg_root, "data", "umls")
    ent_map = os.path.join(data_dir, "entity2id.txt")
    rel_map = os.path.join(data_dir, "relation2id.txt")
    train_path = os.path.join(data_dir, "train.triples")
    dev_path   = os.path.join(data_dir, "dev.triples")
    test_path  = os.path.join(data_dir, "test.triples")

    e_name2id, _ = read_id_map(ent_map)
    r_name2id, _ = read_id_map(rel_map)

    # Load triples (string or id) and map to ids
    train_raw = load_triples_any(train_path)
    dev_raw   = load_triples_any(dev_path)   if os.path.exists(dev_path)  else []
    test_raw  = load_triples_any(test_path)  if os.path.exists(test_path) else []

    train_ids = map_to_ids(train_raw, e_name2id, r_name2id)
    dev_ids   = map_to_ids(dev_raw,   e_name2id, r_name2id) if dev_raw else []
    test_ids  = map_to_ids(test_raw,  e_name2id, r_name2id) if test_raw else []

    # ---- Write CATS training graph for neighbors ----
    out_dir = os.path.join(args.cats_root, args.dataset_dir)
    os.makedirs(out_dir, exist_ok=True)

    # CATS uses `train_full.txt` as the base graph for neighbors
    write_triples(os.path.join(out_dir, "train_full.txt"), train_ids)

    # Valid/test lists (ranking files) were already created by your prepare_umls.py.
    # But we can optionally fill a simple valid set from MHKG dev:
    if args.use_dev_as_valid and dev_ids:
        write_triples(os.path.join(out_dir, "valid.txt"), dev_ids)

    print(f"Wrote {len(train_ids)} triples to {args.dataset_dir}/train_full.txt")
    if args.use_dev_as_valid:
        print(f"Wrote {len(dev_ids)} triples to {args.dataset_dir}/valid.txt")

if __name__ == "__main__":
    main()
