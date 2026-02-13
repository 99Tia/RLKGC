import os, json, argparse, random
from collections import defaultdict

def read_json_or_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        first = f.read(1); f.seek(0)
        if first == '[':
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]

def read_name_id_file(path):
    name2id, id2name = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if '\t' in s:
                a, b = s.split('\t', 1)
            else:
                a, b = s.split(None, 1)
            a, b = a.strip(), b.strip()
            # auto-detect which is the id
            if a.isdigit() and not b.isdigit():
                _id, _name = a, b
            elif b.isdigit() and not a.isdigit():
                _name, _id = a, b
            else:
                # fallback assume a=name, b=id
                _name, _id = a, b
            name2id[_name] = _id
            id2name[_id] = _name
    return name2id, id2name

def read_triples_any(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = s.replace('\t', ' ').split()
            if len(parts) != 3:
                parts = [p.strip() for p in s.split(',')]
            if len(parts) != 3:
                raise ValueError(f"Bad triple line: {line}")
            rows.append(tuple(parts))
    return rows

def map_triples_to_ids(triples, e_name2id, r_name2id):
    out = []
    for a,b,c in triples:
        # already ids?
        if a.isdigit() and b.isdigit() and c.isdigit():
            # try (h,r,t) first; else treat as (h,t,r)
            h, r, t = a, b, c
            if r in r_name2id.values():
                out.append((h, r, t))
            else:
                out.append((h, t, r))
            continue
        # strings: try (h,r,t)
        if (a in e_name2id) and (b in r_name2id) and (c in e_name2id):
            out.append((e_name2id[a], r_name2id[b], e_name2id[c])); continue
        # or (h,t,r)
        if (a in e_name2id) and (b in e_name2id) and (c in r_name2id):
            out.append((e_name2id[a], r_name2id[c], e_name2id[b])); continue
        raise KeyError(f"Cannot map triple {a,b,c} to (h,r,t).")
    return out

def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write('\t'.join(map(str, row)) + '\n')

def build_entity2text(path, id2name):
    rows = [(eid, id2name[eid]) for eid in sorted(id2name.keys(), key=lambda x: int(x))]
    write_tsv(path, rows)

def build_relation2text(path, id2name):
    rows = [(rid, id2name[rid]) for rid in sorted(id2name.keys(), key=lambda x: int(x))]
    write_tsv(path, rows)

def build_train_valid(train_ids, dev_ids, out_dir):
    write_tsv(os.path.join(out_dir, 'train_full.txt'), train_ids + dev_ids)
    write_tsv(os.path.join(out_dir, 'valid.txt'), dev_ids)
    write_tsv(os.path.join(out_dir, 'inductive_graph.txt'), train_ids + dev_ids)

def make_true_lookup(all_triples):
    by_hr, by_rt = defaultdict(set), defaultdict(set)
    for h,r,t in all_triples:
        by_hr[(h,r)].add(t)
        by_rt[(r,t)].add(h)
    return by_hr, by_rt

def sample_negatives(entity_ids, k, forbidden):
    pool = [e for e in entity_ids if e not in forbidden]
    if not pool:
        return [entity_ids[0]] * k
    if len(pool) >= k:
        random.shuffle(pool)
        return pool[:k]
    import random as _rnd
    return [_rnd.choice(pool) for _ in range(k)]

def build_ranking_files(test_ids, all_true_lookup, entity_ids, out_dir, batch_size=50):
    by_hr, by_rt = all_true_lookup
    ent_list = sorted(entity_ids, key=lambda x: int(x))
    NEG = batch_size - 1
    tail_rows, head_rows = [], []

    for h, r, t in test_ids:
        forb_t = by_hr.get((h, r), set())
        neg_tails = sample_negatives(ent_list, NEG, forb_t)
        tail_rows.extend([(h, r, t)] + [(h, r, nt) for nt in neg_tails])

        forb_h = by_rt.get((r, t), set())
        neg_heads = sample_negatives(ent_list, NEG, forb_h)
        head_rows.extend([(h, r, t)] + [(nh, r, t) for nh in neg_heads])

    write_tsv(os.path.join(out_dir, 'ranking_tail.txt'), tail_rows)
    write_tsv(os.path.join(out_dir, 'ranking_head.txt'), head_rows)

def convert_mhkg_jsonl_to_close_path(jsonl_path, out_json_path, e_name2id, r_name2id):
    rows = read_json_or_jsonl(jsonl_path)
    out = defaultdict(list)
    for b in rows:
        # expected MHKG keys: b['query']['e1'], b['candidate']['e2'], b['paths'][i]['steps'] with 'r','e'
        head_name = b['query']['e1']
        tail_name = b['candidate']['e2']
        if head_name not in e_name2id or tail_name not in e_name2id:
            # skip silently instead of crash (safer across datasets)
            continue
        head, tail = e_name2id[head_name], e_name2id[tail_name]
        key = f"{head}-{tail}"

        for ph in b.get('paths', []):
            triples, cur = [], head
            for step in ph.get('steps', []):
                rname = step['r']; ename = step['e']
                if rname not in r_name2id or ename not in e_name2id:
                    triples = []; break
                r = r_name2id[rname]; nxt = e_name2id[ename]
                triples.append([cur, r, nxt]); cur = nxt
            if triples and triples[-1][2] == tail:
                out[key].append(triples)

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['UMLS','Kinship'], help='Target CATS dataset name')
    ap.add_argument('--mhkg_root', required=True, help='Path to MultiHopKG repo root')
    ap.add_argument('--cats_root', required=True, help='Path to CATS repo root')
    ap.add_argument('--mhkg_jsonl', required=True, help='Path to MHKG exported JSON/JSONL with paths')
    ap.add_argument('--test_batch_size', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', choices=['inductive','transductive'], default='inductive')
    ap.add_argument('--train_size', choices=['full','1000','2000'], default='full')
    args = ap.parse_args()
    random.seed(args.seed)

    # ---- infer MHKG subfolder ----
    sub = 'umls' if args.dataset.lower()=='umls' else 'kinship'
    data_dir = os.path.join(args.mhkg_root, 'data', sub)

    e2id_path = os.path.join(data_dir, 'entity2id.txt')
    r2id_path = os.path.join(data_dir, 'relation2id.txt')
    train_path= os.path.join(data_dir, 'train.triples')
    dev_path  = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')

    e_name2id, e_id2name = read_name_id_file(e2id_path)
    r_name2id, r_id2name = read_name_id_file(r2id_path)
    train = read_triples_any(train_path)
    dev   = read_triples_any(dev_path) if os.path.exists(dev_path) else []
    test  = read_triples_any(test_path) if os.path.exists(test_path) else []

    train_ids = map_triples_to_ids(train, e_name2id, r_name2id)
    dev_ids   = map_triples_to_ids(dev,   e_name2id, r_name2id) if dev else []
    test_ids  = map_triples_to_ids(test,  e_name2id, r_name2id) if test else []

    # ---- target dirs in CATS ----
    if args.mode == 'inductive':
        ds_dir    = os.path.join(args.cats_root, 'datasets', f'{args.dataset}-inductive')
        paths_dir = os.path.join(ds_dir, 'paths')
        out_json  = os.path.join(paths_dir, 'close_path.json')
    else:
        ds_dir    = os.path.join(args.cats_root, 'datasets', f'{args.dataset}')
        paths_dir = os.path.join(ds_dir, 'paths')
        out_json  = os.path.join(paths_dir, f'close_path_train_size_{args.train_size}.json')

    os.makedirs(paths_dir, exist_ok=True)

    # ---- write texts/maps ----
    build_entity2text(os.path.join(ds_dir, 'entity2text.txt'), e_id2name)
    build_relation2text(os.path.join(ds_dir, 'relation2text.txt'), r_id2name)

    # ---- graph files ----
    build_train_valid(train_ids, dev_ids, ds_dir)

    # ---- ranking files for prediction ----
    all_true_lookup = make_true_lookup(train_ids + dev_ids + test_ids)
    all_entity_ids = list(e_id2name.keys())
    build_ranking_files(test_ids, all_true_lookup, all_entity_ids, ds_dir, batch_size=args.test_batch_size)

    # ---- paths from MHKG JSON/JSONL ----
    convert_mhkg_jsonl_to_close_path(args.mhkg_jsonl, out_json, e_name2id, r_name2id)

    print(f"✔ Wrote CATS dataset to: {ds_dir}")
    print("   - entity2text.txt, relation2text.txt")
    print("   - train_full.txt, valid.txt, inductive_graph.txt")
    print("   - ranking_head.txt, ranking_tail.txt")
    print(f"   - {os.path.relpath(out_json, ds_dir)}")

if __name__ == '__main__':
    main()
