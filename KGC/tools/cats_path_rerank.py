import os, json, argparse, sys
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data_manager import DataManager
from prompt_templates import SUBGRAPH_REASON_PROMPT

def has_all_ids(dm, path):
    for h, r, t in path:
        if str(h) not in dm.entity2text: return False
        if str(t) not in dm.entity2text: return False
        if str(r) not in dm.relation2text: return False
    return True

def filter_ok_paths(dm, id_paths):
    ok = []
    for p in id_paths:
        p = [[str(h), str(r), str(t)] for (h,r,t) in p]
        if has_all_ids(dm, p):
            ok.append(p)
    return ok


def triple_to_sentence_safe(dm, tri):
    h, r, t = map(str, tri)
    return f"('{dm.entity2text.get(h, f'<E:{h}>')}' {dm.relation2text.get(r, f'<R:{r}>')} '{dm.entity2text.get(t, f'<E:{t}>')}')"

def render_paths_safe(dm, id_paths, dataset="UMLS"):
    lines = []
    for path in id_paths:
        seq = " -> ".join(triple_to_sentence_safe(dm, tri, dataset) for tri in path)
        lines.append(seq)
    return "\n".join(lines)

def cal_logodds(model, tokenizer, generation_config, prompts: List[str], device="cuda"):
    """
    Run a single-token generation and return SR log-odds for 'Y' vs 'N' for each prompt.
    """
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    out = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_config
    )
    logits = out.scores[0]
    logp = torch.log_softmax(logits, dim=-1)
    y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    n_id = tokenizer.encode("N", add_special_tokens=False)[0]
    return (logp[:, y_id] - logp[:, n_id]).tolist()  # log-odds


def render_paths(dm: DataManager, id_paths):
    lines = []
    for path in id_paths:
        seq = " -> ".join(triple_to_sentence_safe(dm, (h, r, t)) for (h, r, t) in path)
        lines.append(seq)
    return "\n".join(lines)

def build_sr_prompt(dm, triple, neighbors, id_paths):
    return SUBGRAPH_REASON_PROMPT.format(
        neighbor_triples="\n".join(neighbors),
        reasoning_paths=render_paths(dm, id_paths),
        test_triple=triple_to_sentence_safe(dm, triple)
    )

def load_name_id_maps(dm: DataManager):
    e_id2name = dm._load_text_file("entity2text.txt")
    r_id2name = dm._load_text_file("relation2text.txt")
    e_name2id = {v: k for k, v in e_id2name.items()}
    r_name2id = {v: k for k, v in r_id2name.items()}
    return e_name2id, r_name2id, e_id2name, r_id2name


def mhkg_iter(mhkg_path: str):
    def read_any(p):
        with open(p, 'r', encoding='utf-8') as f:
            first = f.read(1); f.seek(0)
            if first == '[':
                return json.load(f)
            return [json.loads(line) for line in f if line.strip()]
    rows = read_any(mhkg_path)
    for obj in rows:
        q = obj.get("query", {})
        c = obj.get("candidate", {})
        yield {
            "h_name": q.get("e1"),
            "r_name": q.get("r"),
            "t_name": c.get("e2"),
            "h_id": str(q.get("e1_id","")),
            "r_id": str(q.get("r_id","")),
            "t_id": str(c.get("e2_id","")),
            "paths": obj.get("paths", [])
        }


def closepath_iter(close_path_json: str, queries_path: str, e_id2name: Dict[str, str], r_id2name: Dict[str, str]):
    """
    Iterate (h_name, r_name, t_name, id_paths) using close_path.json and a queries file (triples).
    The queries file can be a 3-column TSV/space-delimited 'head relation tail' with NAMES.
    """
    paths_dict = json.load(open(close_path_json, 'r', encoding='utf-8')) 
    e_name2id = {v: k for k, v in e_id2name.items()}
    r_name2id = {v: k for k, v in r_id2name.items()}

    def parse_triples(p):
        out = []
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s: continue
                parts = s.split('\t')
                if len(parts) != 3:
                    parts = s.split()
                if len(parts) != 3:
                    raise ValueError(f"Bad triple line: {line}")
                out.append(tuple(parts))
        return out

    triples = parse_triples(queries_path)
    for (h_name, r_name, t_name) in triples:
        if h_name not in e_name2id or t_name not in e_name2id or r_name not in r_name2id:
            continue
        h, r, t = e_name2id[h_name], r_name2id[r_name], e_name2id[t_name]
        key = f"{h}-{t}"
        id_paths = paths_dict.get(key, [])
        yield h_name, r_name, t_name, id_paths  

def id_paths_from_mhkg_steps(h_id: str, t_id: str, steps_list, e_name2id, r_name2id):
    """
    Convert MHKG 'steps' (with names) to ID path [[[h,r,t],...], ...], keeping only paths ending in t_id.
    """
    id_paths = []
    for ph in steps_list:
        triples = []
        cur = h_id
        ok = True
        for step in ph.get("steps", []):
            rname = step["r"]; ename = step["e"]
            if rname not in r_name2id or ename not in e_name2id:
                ok = False; break
            rid = r_name2id[rname]
            nid = e_name2id[ename]
            triples.append([cur, rid, nid])
            cur = nid
        if ok and triples and triples[-1][2] == t_id:
            id_paths.append(triples)
    return id_paths

def id_paths_from_steps_using_ids(head_id: str, tail_id: str, paths_steps) -> List[List[List[str]]]:
    """Prefer per-step r_id/e_id; fall back to names if IDs missing (handled elsewhere)."""
    id_paths = []
    for ph in paths_steps:
        triples = []
        cur = head_id
        ok = True
        for st in ph.get("steps", []):
            rid = st.get("r_id")
            nid = st.get("e_id")
            if rid is None or nid is None:
                ok = False
                break
            rid = str(rid); nid = str(nid)
            triples.append([cur, rid, nid])
            cur = nid
        if ok and triples and triples[-1][2] == tail_id:
            id_paths.append(triples)
    return id_paths


def main():
    ap = argparse.ArgumentParser(description="Path reranking with CATS SR (add-one + removal gains)")
    ap.add_argument("--dataset", default="UMLS", choices=["FB15k-237-subset", "NELL-995-subset", "WN18RR-subset", "UMLS"])
    ap.add_argument("--setting", default="inductive", choices=["inductive", "transductive"])
    ap.add_argument("--train_size", default="full", choices=["full", "1000", "2000"])
    ap.add_argument("--model_name", required=True, help="Path to SFT model or HF model id")
    ap.add_argument("--llm_type", default="sft", choices=["sft","base"])
    ap.add_argument("--device", default="cuda", help="cuda / cuda:0 / cuda:1 / cpu")

  
    ap.add_argument("--mode", choices=["mhkg", "closepath"], default="mhkg",
                    help="mhkg: iterate MHKG outputs; closepath: use close_path + queries")
    ap.add_argument("--mhkg_json", help="Path to outputs/*_cats.json or .jsonl from MultiHopKG")
    ap.add_argument("--close_path_json", help="Path to datasets/.../paths/close_path*.json")
    ap.add_argument("--queries_file", help="Queries file with (h r t) names, e.g., test.triples")

 
    ap.add_argument("--alpha", type=float, default=0.7, help="weight for add-one gain")
    ap.add_argument("--beta", type=float, default=0.3, help="weight for removal gain")
    ap.add_argument("--hops_lambda", type=float, default=0.0, help="optional penalty per hop")
    ap.add_argument("--outfile", required=True, help="Where to write JSONL with reranked paths")

    args = ap.parse_args()

    
    dm = DataManager(dataset=args.dataset, setting=args.setting, train_size=args.train_size,
                     model_name=args.model_name, llm_type=args.llm_type)
    e_name2id, r_name2id, e_id2name, r_id2name = load_name_id_maps(dm)

    tok = AutoTokenizer.from_pretrained(dm.model_path)
    model = AutoModelForCausalLM.from_pretrained(dm.model_path, torch_dtype="auto",
                                                 device_map="auto" if args.device.startswith("cuda") else None).to(args.device)

    gen_cfg = dict(temperature=0, top_k=0, top_p=0, do_sample=False, max_new_tokens=1)

    if args.mode == "mhkg":
        if not args.mhkg_json:
            sys.exit("ERROR: --mhkg_json is required in mhkg mode")
        iterator = mhkg_iter(args.mhkg_json)
        def id_paths_builder(rec):
            if rec["h_id"] and rec["t_id"]:
                id_paths = id_paths_from_steps_using_ids(rec["h_id"], rec["t_id"], rec["paths"])
                if id_paths:
                    return rec["h_id"], rec["r_id"], rec["t_id"], id_paths
            h_name, r_name, t_name = rec["h_name"], rec["r_name"], rec["t_name"]
            if not (h_name in e_name2id and t_name in e_name2id and r_name in r_name2id):
                return None
            h_id, r_id, t_id = e_name2id[h_name], r_name2id[r_name], e_name2id[t_name]
            id_paths = id_paths_from_mhkg_steps(h_id, t_id, rec["paths"], e_name2id, r_name2id)
            if not id_paths:
                return None
            return h_id, r_id, t_id, id_paths

    else:
        if not (args.close_path_json and args.queries_file):
            sys.exit("ERROR: --close_path_json and --queries_file are required in closepath mode")
        iterator = closepath_iter(args.close_path_json, args.queries_file, e_id2name, r_id2name)
        def id_paths_builder(h_name, r_name, t_name, id_paths_input):
            return id_paths_input

    n_total = 0
    n_scored = 0
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    
    with open(args.outfile, "w", encoding="utf-8") as out:
        for rec in iterator:
            n_total += 1
            built = id_paths_builder(rec)
            if not built:
                continue
            h_id, r_id, t_id, id_paths = built
            h_name, r_name, t_name = rec["h_name"], rec["r_name"], rec["t_name"]

            
            triple_ids = (str(h_id), str(r_id), str(t_id))
            if (triple_ids[0] in dm.entity2text and 
                triple_ids[1] in dm.relation2text and
                triple_ids[2] in dm.entity2text):
                neighbors = dm.neighbor_triple_finder(triple_ids)
            else:
                neighbors = []  

            def sr_prompt(paths):
                return build_sr_prompt(dm, triple_ids, neighbors, paths)

            prompts = [sr_prompt([]), sr_prompt(id_paths)]
            prompts += [sr_prompt([p]) for p in id_paths]
            prompts += [sr_prompt(id_paths[:i] + id_paths[i+1:]) for i in range(len(id_paths))]

            logs = cal_logodds(model, tok, gen_cfg, prompts, device=args.device)
            lN = logs[0]; lALL = logs[1]
            add = logs[2 : 2 + len(id_paths)]
            rem = logs[2 + len(id_paths) : 2 + 2*len(id_paths)]

            scored_paths = []
            for i, p in enumerate(id_paths):
                add_gain = add[i] - lN
                rem_gain = (lALL - rem[i]) if i < len(rem) else 0.0
                hops = len(p)
                score = args.alpha * add_gain + args.beta * rem_gain - args.hops_lambda * hops
                p_named = [[dm.entity2text.get(str(a), f"<E:{a}>"), dm.relation2text.get(str(b), f"<R:{b}>"), dm.entity2text.get(str(c), f"<E:{c}>"),] for (a,b,c) in p]
                scored_paths.append({
                    "score": float(score),
                    "add_gain": float(add_gain),
                    "rem_gain": float(rem_gain),
                    "hops": hops,
                    "path": p_named
                    })

            scored_paths.sort(key=lambda x: x["score"], reverse=True)
            out.write(json.dumps({
                "head": h_name,
                "relation": r_name,
                "tail": t_name,
                "sr_logodds_all": float(lALL),
                "paths_ranked": scored_paths
            }) + "\n")
            n_scored += 1


    print(f"Done. Scored {n_scored}/{n_total} queries. Wrote: {args.outfile}")

if __name__ == "__main__":
    main()

