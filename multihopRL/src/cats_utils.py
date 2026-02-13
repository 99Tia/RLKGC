# src/cats_utils.py
from collections import deque
import math
import torch

# ---------- helpers to keep PN state & device consistent ----------

def _device_from(pn, kg):
    """
    Ensure all tensors are created on the same device as the policy network (PN)
    or, if PN has no params, fall back to KG embeddings device, else CPU.
    """
    try:
        return next(pn.parameters()).device
    except StopIteration:
        pass
    if hasattr(kg, "entity_embeddings"):
        try:
            return kg.entity_embeddings.weight.device
        except Exception:
            pass
    return torch.device("cpu")

def _ltensor(x, device):
    """1-D long tensor [x] on device."""
    return torch.tensor([x], dtype=torch.long, device=device)

def _init_path(pn, kg, e_s, device):
    """Initialize PN path LSTM with (START_R, e_s) for batch=1."""
    r_start = torch.tensor([kg.dummy_start_r], dtype=torch.long, device=device)
    e_start = torch.tensor([e_s],           dtype=torch.long, device=device)
    pn.initialize_path((r_start, e_start), kg)

def _replay_path(pn, kg, path, device):
    """
    Rebuild PN path history for a single path:
    path = [(START_R, e_s), (r1, e1), (r2, e2), ...]
    """
    _init_path(pn, kg, path[0][1], device)
    for t in range(1, len(path)):
        r_t = torch.tensor([path[t][0]], dtype=torch.long, device=device)
        e_t = torch.tensor([path[t][1]], dtype=torch.long, device=device)
        pn.update_path((r_t, e_t), kg, offset=None)

# ---------- bounded BFS guided by PN, to a fixed tail e* ----------

def bounded_bfs_to_target(pn, kg, e_s, q, e_star, max_depth=3, branch_cap=10, max_results=100):
    """
    Return paths that all END at e_star.
    Each path is a list of (r_id, e_id), starting with (START_R, e_s).
    BFS is guided by PN probabilities and pruned to top-`branch_cap` at each node.
    """
    device = _device_from(pn, kg)
    START_R = kg.dummy_start_r

    start_path = [(START_R, e_s)]
    results, Q = [], deque([start_path])

    while Q:
        path = Q.popleft()
        depth = len(path) - 1
        if depth >= max_depth:
            continue

        # Rebuild PN history to THIS path so H matches the path
        _replay_path(pn, kg, path, device)

        last_r, cur_e = path[-1]

        # Build 1-item batch obs for PN at current node
        e_batch   = torch.tensor([cur_e], dtype=torch.long, device=device)
        e_s_t     = _ltensor(e_s, device)
        q_t       = _ltensor(q, device)
        e_t_t     = _ltensor(e_star, device)               # chosen tail
        last_r_t  = _ltensor(last_r, device)
        last_step = (depth + 1 == max_depth)
        # seen_nodes: minimal stub (PN only expects shape/type)
        seen_nodes= torch.full((1, 1), fill_value=kg.dummy_e, dtype=torch.long, device=device)

        obs = [e_s_t, q_t, e_t_t, last_step, last_r_t, seen_nodes]

        # Use BUCKETING (so we don't require kg.action_space)
        db_outcomes, _, _ = pn.transit(
            e_batch, obs, kg,
            use_action_space_bucketing=True,
            merge_aspace_batching_outcome=True,
        )
        # After merging, one tuple in db_outcomes:
        (r_space, e_space), action_mask = db_outcomes[0][0]  # shapes: [1, num_actions]
        action_dist = db_outcomes[0][1][0]                   # [num_actions]

        # Collect valid actions
        cand = []
        for idx in range(action_dist.size(0)):
            valid = action_mask[0, idx]
            try:
                valid = bool(int(valid))
            except Exception:
                valid = bool(valid)
            if not valid:
                continue

            r_next = int(r_space[0, idx])
            e_next = int(e_space[0, idx])
            if e_next == kg.dummy_e:
                continue
            # avoid cycles
            if any(e_next == ent for (_, ent) in path):
                continue

            p = float(action_dist[idx])
            cand.append((p, r_next, e_next))

        # policy-guided pruning
        cand.sort(key=lambda x: x[0], reverse=True)
        cand = cand[:branch_cap]

        for p, r_next, e_next in cand:
            new_path = path + [(r_next, e_next)]
            if e_next == e_star:
                results.append(new_path)
                if len(results) >= max_results:
                    return results
            else:
                Q.append(new_path)

    return results

def score_path_logprob(pn, kg, e_s, q, path):
    """
    Sum log π(a_t|s_t) along the path (t = 1..T) with correct PN state progression.
    Path format: [(START_R, e_s), (r1, e1), (r2, e2), ...]
    """

    device = _device_from(pn, kg)

    # Start PN history at (START_R, e_s)
    _init_path(pn, kg, e_s, device)
    total = 0.0

    for t in range(1, len(path)):
        last_r, cur_e = path[t - 1]
        r_next, e_next = path[t]

        # Build obs for the *current* node (before taking (r_next, e_next))
        e_batch   = torch.tensor([cur_e], dtype=torch.long, device=device)
        e_s_t     = _ltensor(e_s, device)
        q_t       = _ltensor(q, device)
        e_t_t     = _ltensor(e_next, device)          # we want prob of going to e_next now
        last_r_t  = _ltensor(last_r, device)
        last_step = (t == len(path) - 1)

        # Use real visited nodes so PN’s masking (if enabled) is consistent.
        # If this ever causes a shape warning in your fork, replace with the stub line below.
        seen_nodes = torch.tensor([[p[1] for p in path[:t]]], dtype=torch.long, device=device)
        # Fallback stub (uncomment if needed):
        # seen_nodes = torch.full((1, 1), fill_value=kg.dummy_e, dtype=torch.long, device=device)

        obs = [e_s_t, q_t, e_t_t, last_step, last_r_t, seen_nodes]

        db_outcomes, _, _ = pn.transit(
            e_batch, obs, kg,
            use_action_space_bucketing=True,
            merge_aspace_batching_outcome=True,
        )
        (r_space, e_space), action_mask = db_outcomes[0][0]
        action_dist = db_outcomes[0][1][0]  # [num_actions]

        # Grab π(r_next, e_next | state_t)
        step_logp = float("-inf")
        for idx in range(action_dist.size(0)):
            if not bool(int(action_mask[0, idx])):
                continue
            if int(r_space[0, idx]) == r_next and int(e_space[0, idx]) == e_next:
                p = float(action_dist[idx])
                step_logp = float("-inf") if p <= 0.0 else math.log(p)
                break

        total += step_logp

        # Advance PN history by taking (r_next, e_next)
        pn.update_path((_ltensor(r_next, device), _ltensor(e_next, device)), kg, offset=None)

    return total

# ---------- pretty export ----------

def format_path_for_export(path, kg, logprob=None):
    """
    Convert list of (r_id, e_id) to readable dict with names.
    """
    steps = []
    for t in range(1, len(path)):
        r_id, e_id = path[t]
        steps.append({
            'r_id': int(r_id), 'r': kg.id2relation[int(r_id)],
            'e_id': int(e_id), 'e': kg.id2entity[int(e_id)],
        })
    out = {'steps': steps}
    if logprob is not None:
        out['logprob'] = float(logprob)
    return out
