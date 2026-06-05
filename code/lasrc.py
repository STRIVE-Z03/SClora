"""
LASRC - Layer-Adaptive Stabilized Residual Composition.

Core operator module for open-pool LoRA composition.
Replaces the globally-uniform linear sum with a layer-adaptive,
residual-orthogonal composition operator.

Design:
  - precompute()        : call ONCE per support group (before Nevergrad loop)
  - compose()           : call EACH Nevergrad step (weight-dependent)
  - compose_final()     : call ONCE after optimisation to set model weights
"""

import math
import re
from collections import defaultdict

import torch

DEFAULT_GAMMA = 0.5
DEFAULT_GAMMA_MODE = "overlap"
DEFAULT_GAMMA_FLOOR = 0.05
DEFAULT_NORM_GUARD = 0.3
DEFAULT_CONSENSUS_ALPHA = 1.0
DEFAULT_ALIGNMENT_ALPHA = 0.0


# ---------------------------------------------------------------------------
# 1. Layer grouping
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"((?:encoder|decoder)\.block\.(\d+))")


def _parse_block_id(key):
    """Extract transformer block id from a LoRA state_dict key.

    Examples
    --------
    'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.weight'
    -> 'encoder.block.0'
    """
    m = _BLOCK_RE.search(key)
    return m.group(1) if m else "_other"


def group_keys_by_layer(keys):
    """Group state_dict keys by transformer block.

    Returns
    -------
    dict : {block_id: [key, ...]}
    """
    groups = defaultdict(list)
    for key in keys:
        groups[_parse_block_id(key)].append(key)
    return dict(groups)


# ---------------------------------------------------------------------------
# 2. Per-layer overlap analysis  (pre-computed once per support group)
# ---------------------------------------------------------------------------

def compute_layerwise_overlap(cache, lora_module_list, layer_groups):
    """Compute mean pairwise cosine overlap per layer group.

    Returns
    -------
    overlap_stats   : {block_id: {'mean_overlap': float, 'max_overlap': float}}
    expert_norms    : {block_id: [norm_0, norm_1, ...]}   (per-expert L2 norm)
    expert_flat_vecs: {block_id: [Tensor, ...]}  (flat expert vectors, reusable)
    """
    N = len(lora_module_list)
    overlap_stats = {}
    expert_norms = {}
    expert_flat_vecs = {}

    for block_id, keys in layer_groups.items():
        vecs = []
        norms = []
        for module_id in lora_module_list:
            sd = cache[module_id]
            parts = [sd[k].float().flatten() for k in keys]
            vec = torch.cat(parts)
            vecs.append(vec)
            norms.append(vec.norm().item())
        expert_norms[block_id] = norms
        expert_flat_vecs[block_id] = vecs

        if N < 2:
            overlap_stats[block_id] = {"mean_overlap": 0.0, "max_overlap": 0.0}
            continue

        mat = torch.stack(vecs)                                     # (N, D)
        row_norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        mat_n = mat / row_norms
        cos_sim = mat_n @ mat_n.T                                   # (N, N)

        tri = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        pairwise = cos_sim[tri].abs()
        mean_ov = pairwise.mean().item() if pairwise.numel() > 0 else 0.0
        max_ov = pairwise.max().item() if pairwise.numel() > 0 else 0.0
        overlap_stats[block_id] = {"mean_overlap": mean_ov, "max_overlap": max_ov}

    return overlap_stats, expert_norms, expert_flat_vecs


# ---------------------------------------------------------------------------
# 3. Budget scheduling  (pre-computed once)
# ---------------------------------------------------------------------------

def schedule_layer_budgets(overlap_stats, k_min=1, k_max=8, beta=1.5):
    """Assign expert budget per layer: higher overlap -> sparser budget."""
    budgets = {}
    for block_id, stats in overlap_stats.items():
        overlap = max(0.0, min(1.0, float(stats["mean_overlap"])))
        span = max(int(k_max) - int(k_min), 0)
        raw = int(k_min + round(span * math.exp(-float(beta) * overlap)))
        budgets[block_id] = max(k_min, min(k_max, raw))
    return budgets


# ---------------------------------------------------------------------------
# 4. Pre-computation entry point
# ---------------------------------------------------------------------------

def precompute(cache, lora_module_list, k_min=1, k_max=8, beta=1.5):
    """Pre-compute everything that does NOT depend on Nevergrad weights.

    Call once per support group before the optimisation loop.
    """
    keys = list(cache[lora_module_list[0]].keys())
    layer_groups = group_keys_by_layer(keys)
    overlap_stats, expert_norms, expert_flat_vecs = compute_layerwise_overlap(
        cache, lora_module_list, layer_groups,
    )
    layer_budgets = schedule_layer_budgets(
        overlap_stats, k_min=k_min, k_max=k_max, beta=beta,
    )
    # pre-compute per-block key sizes for fast split during compose
    first_module = lora_module_list[0]
    block_key_meta = {}
    for block_id, bkeys in layer_groups.items():
        sizes = [cache[first_module][k].numel() for k in bkeys]
        shapes = [cache[first_module][k].shape for k in bkeys]
        block_key_meta[block_id] = {"keys": bkeys, "sizes": sizes, "shapes": shapes}

    mean_overlaps = [stats["mean_overlap"] for stats in overlap_stats.values()]
    min_mean_overlap = min(mean_overlaps) if mean_overlaps else 0.0
    max_mean_overlap = max(mean_overlaps) if mean_overlaps else 0.0

    return {
        "layer_groups": layer_groups,
        "overlap_stats": overlap_stats,
        "layer_budgets": layer_budgets,
        "expert_norms": expert_norms,
        "block_key_meta": block_key_meta,
        "expert_flat_vecs": expert_flat_vecs,
        "min_mean_overlap": min_mean_overlap,
        "max_mean_overlap": max_mean_overlap,
    }


# ---------------------------------------------------------------------------
# 5. Layer mask construction  (weight-dependent, per Nevergrad step)
# ---------------------------------------------------------------------------

def build_layer_masks(weights, expert_norms, layer_budgets, layer_groups, mask_mode="dense"):
    """Per-layer binary expert mask: top-k_l by |alpha_i| * ||DeltaW_i^l||."""
    N = len(weights)
    if mask_mode == "dense":
        return {block_id: [True] * N for block_id in layer_groups}
    masks = {}
    for block_id in layer_groups:
        budget = min(layer_budgets.get(block_id, N), N)
        norms = expert_norms.get(block_id, [1.0] * N)
        scores = [abs(float(weights[i])) * norms[i] for i in range(N)]
        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
        active = set(ranked[:budget])
        masks[block_id] = [i in active for i in range(N)]
    return masks


# ---------------------------------------------------------------------------
# 6. Residual-orthogonal composition  (weight-dependent)
# ---------------------------------------------------------------------------

def _expert_order(weights, norms_block):
    """Importance-sorted expert indices (descending)."""
    N = len(weights)
    scores = [abs(float(weights[i])) * norms_block[i] for i in range(N)]
    return sorted(range(N), key=lambda i: scores[i], reverse=True)


def _resolve_layer_gamma(
    precomputed,
    block_id,
    gamma=DEFAULT_GAMMA,
    gamma_mode=DEFAULT_GAMMA_MODE,
    gamma_floor=DEFAULT_GAMMA_FLOOR,
):
    gamma = float(gamma)
    if gamma <= 0.0:
        return 0.0
    if gamma_mode != "overlap":
        return gamma

    overlap_stats = precomputed.get("overlap_stats", {})
    mean_overlap = float(overlap_stats.get(block_id, {}).get("mean_overlap", 0.0))
    min_mean_overlap = float(precomputed.get("min_mean_overlap", 0.0))
    max_mean_overlap = float(precomputed.get("max_mean_overlap", 0.0))
    if max_mean_overlap - min_mean_overlap < 1e-8:
        normalized = 1.0
    else:
        normalized = (mean_overlap - min_mean_overlap) / (max_mean_overlap - min_mean_overlap)
    normalized = max(0.0, min(1.0, normalized))
    gamma_floor = max(0.0, min(float(gamma_floor), gamma))
    return gamma_floor + (gamma - gamma_floor) * normalized


def _apply_norm_guard(gamma_value, linear_norm, residual_norm, norm_guard=0.0):
    gamma_value = float(gamma_value)
    norm_guard = float(norm_guard)
    if gamma_value <= 0.0 or norm_guard <= 0.0:
        return gamma_value
    if linear_norm <= 1e-9:
        return gamma_value

    ratio = residual_norm / linear_norm
    lower = max(1e-6, min(norm_guard, 1.0))
    upper = 1.0 / lower
    if ratio < lower:
        gamma_value = gamma_value * max(0.0, min(1.0, ratio / lower))
    elif ratio > upper:
        gamma_value = gamma_value * max(0.0, min(1.0, upper / ratio))
    return gamma_value


def _apply_consensus_scaling(
    gamma_value,
    linear_norm,
    residual_norm,
    consensus_alpha=DEFAULT_CONSENSUS_ALPHA,
):
    """Reduce gamma for layers where orthogonalization removed substantial shared signal.

    survival = residual_norm / linear_norm measures how much of the original
    energy survived orthogonalization.  Low survival means experts shared a
    large consensus component that was projected out.  For tasks that rely on
    global knowledge coverage (disambiguation, factual recall, fine-grained
    classification), this consensus is critical.

    gamma_value *= survival ** consensus_alpha

    Parameters
    ----------
    consensus_alpha : float
        0 = disabled, 1.0 = linear survival scaling, 2.0 = aggressive protection.
    """
    gamma_value = float(gamma_value)
    consensus_alpha = float(consensus_alpha)
    if gamma_value <= 0.0 or consensus_alpha <= 0.0 or linear_norm <= 1e-9:
        return gamma_value
    survival = residual_norm / linear_norm
    survival = max(0.0, min(1.0, survival))
    return gamma_value * (survival ** consensus_alpha)


def _safe_cosine(vec_a, vec_b, norm_a=None, norm_b=None):
    if norm_a is None:
        norm_a = vec_a.norm().item()
    if norm_b is None:
        norm_b = vec_b.norm().item()
    norm_a = float(norm_a)
    norm_b = float(norm_b)
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 1.0
    cosine = torch.dot(vec_a, vec_b).item() / (norm_a * norm_b)
    return max(-1.0, min(1.0, cosine))


def _apply_alignment_scaling(
    gamma_value,
    alignment,
    alignment_alpha=DEFAULT_ALIGNMENT_ALPHA,
):
    """Reduce gamma when residual direction diverges from the linear merge.

    The cosine alignment is mapped from [-1, 1] to [0, 1] before exponentiation:

        alignment_scale = ((alignment + 1) / 2) ** alignment_alpha

    This keeps fully aligned residuals unchanged, halves gamma for orthogonal
    residuals when alpha=1, and drives gamma to zero for opposite directions.
    """
    gamma_value = float(gamma_value)
    alignment_alpha = float(alignment_alpha)
    if gamma_value <= 0.0 or alignment_alpha <= 0.0:
        return gamma_value
    alignment = max(-1.0, min(1.0, float(alignment)))
    alignment_scale = max(0.0, min(1.0, 0.5 * (alignment + 1.0)))
    return gamma_value * (alignment_scale ** alignment_alpha)


@torch.no_grad()
def compose_state_dict(
    cache, lora_module_list, weights,
    precomputed, layer_masks, prune_threshold=0.05,
    gamma=DEFAULT_GAMMA,
    gamma_mode=DEFAULT_GAMMA_MODE,
    gamma_floor=DEFAULT_GAMMA_FLOOR,
    norm_guard=DEFAULT_NORM_GUARD,
    consensus_alpha=DEFAULT_CONSENSUS_ALPHA,
    alignment_alpha=DEFAULT_ALIGNMENT_ALPHA,
):
    """Residual-orthogonal merge using Modified Gram-Schmidt per block.

    **Scale-preserving**: residual is rescaled to match linear norm before
    interpolation, so gamma only changes *direction* (de-redundancy) without
    losing *energy*.  This is critical to beat the linear-sum baseline.

    **Consensus-aware**: layers where orthogonalization removed a large shared
    component automatically get reduced gamma, protecting tasks that depend on
    global knowledge coverage and disambiguation.

    **Alignment-aware**: when the residual direction drifts too far from the
    linear merge direction, gamma is reduced again based on cosine alignment.
    This protects fine-grained tasks that are sensitive to large directional
    rotations even when residual energy is preserved.

    Returns
    -------
    final_state_dict : dict
    layer_stats      : list of per-block diagnostic dicts
    """
    N = len(lora_module_list)
    layer_groups = precomputed["layer_groups"]
    expert_norms = precomputed["expert_norms"]
    block_key_meta = precomputed["block_key_meta"]
    expert_flat_vecs = precomputed["expert_flat_vecs"]

    final_state_dict = {}
    layer_stats = []

    for block_id, keys in layer_groups.items():
        mask = layer_masks.get(block_id, [True] * N)
        norms_block = expert_norms.get(block_id, [1.0] * N)
        order = _expert_order(weights, norms_block)
        meta = block_key_meta[block_id]
        key_sizes = meta["sizes"]
        key_shapes = meta["shapes"]
        block_vecs = expert_flat_vecs[block_id]  # pre-computed flat vecs

        bases = []
        linear_accum = None
        residual_accum = None
        block_active = 0
        block_pruned = 0

        for idx in order:
            if not mask[idx]:
                continue
            w_i = float(weights[idx])
            if abs(w_i) < 1e-9:
                continue

            raw_flat = w_i * block_vecs[idx]  # use pre-computed vec

            if linear_accum is None:
                linear_accum = raw_flat.clone()
            else:
                linear_accum = linear_accum + raw_flat

            residual = raw_flat.clone()
            for b in bases:
                coeff = torch.dot(residual, b)
                residual = residual - coeff * b

            raw_norm = raw_flat.norm().item()
            res_norm = residual.norm().item()

            if raw_norm > 1e-9 and (res_norm / raw_norm) < prune_threshold:
                block_pruned += 1
                continue

            if residual_accum is None:
                residual_accum = residual
            else:
                residual_accum = residual_accum + residual
            block_active += 1

            if res_norm > 1e-9:
                bases.append(residual / res_norm)

        if linear_accum is None:
            linear_accum = torch.zeros(sum(key_sizes), dtype=torch.float32)
        if residual_accum is None:
            residual_accum = torch.zeros_like(linear_accum)

        linear_norm = linear_accum.norm().item()
        residual_norm = residual_accum.norm().item()
        survival_ratio = (residual_norm / linear_norm) if linear_norm > 1e-9 else 0.0

        # ---- Scale-preserving rescale ----
        # Rescale residual to match linear norm so interpolation only
        # rotates the direction without shrinking magnitude.
        if residual_norm > 1e-9 and linear_norm > 1e-9:
            residual_scaled = residual_accum * (linear_norm / residual_norm)
            alignment = _safe_cosine(
                linear_accum,
                residual_scaled,
                norm_a=linear_norm,
                norm_b=linear_norm,
            )
        else:
            residual_scaled = residual_accum
            alignment = 1.0

        gamma_value = _resolve_layer_gamma(
            precomputed,
            block_id,
            gamma=gamma,
            gamma_mode=gamma_mode,
            gamma_floor=gamma_floor,
        )
        gamma_value = _apply_norm_guard(
            gamma_value,
            linear_norm,
            residual_norm,
            norm_guard=norm_guard,
        )
        gamma_after_norm_guard = gamma_value
        gamma_value = _apply_consensus_scaling(
            gamma_value,
            linear_norm,
            residual_norm,
            consensus_alpha=consensus_alpha,
        )
        gamma_after_consensus = gamma_value
        gamma_value = _apply_alignment_scaling(
            gamma_value,
            alignment,
            alignment_alpha=alignment_alpha,
        )
        final_flat = (1.0 - gamma_value) * linear_accum + gamma_value * residual_scaled

        offset = 0
        for key, size, shape in zip(keys, key_sizes, key_shapes):
            final_state_dict[key] = final_flat[offset: offset + size].view(shape)
            offset += size

        layer_stats.append({
            "block_id": block_id,
            "budget": sum(mask),
            "active_experts": block_active,
            "pruned_residuals": block_pruned,
            "num_bases": len(bases),
            "gamma": gamma_value,
            "gamma_pre_consensus": gamma_after_norm_guard,
            "gamma_pre_alignment": gamma_after_consensus,
            "linear_norm": linear_norm,
            "residual_norm": residual_norm,
            "survival_ratio": survival_ratio,
            "consensus_scale": (
                gamma_after_consensus / gamma_after_norm_guard
                if gamma_after_norm_guard > 1e-9
                else 1.0
            ),
            "alignment": alignment,
            "alignment_scale": (
                gamma_value / gamma_after_consensus
                if gamma_after_consensus > 1e-9
                else 1.0
            ),
            "final_norm": final_flat.norm().item(),
        })

    return final_state_dict, layer_stats


# ---------------------------------------------------------------------------
# 7. Coverage regularization
# ---------------------------------------------------------------------------

def coverage_regularization(layer_masks, lam=0.01):
    """Negative-entropy penalty encouraging diverse expert usage across layers."""
    if lam <= 0 or not layer_masks:
        return 0.0

    sample = next(iter(layer_masks.values()))
    N = len(sample)
    if N <= 1:
        return 0.0

    usage = [0.0] * N
    for m in layer_masks.values():
        for i, active in enumerate(m):
            if active:
                usage[i] += 1.0

    total = sum(usage)
    if total < 1e-9:
        return 0.0

    entropy = 0.0
    for u in usage:
        p = u / total
        if p > 1e-12:
            entropy -= p * math.log(p)

    max_entropy = math.log(N)
    if max_entropy < 1e-12:
        return 0.0
    return lam * (1.0 - entropy / max_entropy)


# ---------------------------------------------------------------------------
# 8. Full pipeline  (called per Nevergrad step)
# ---------------------------------------------------------------------------

def compose(cache, lora_module_list, weights, precomputed,
            prune_threshold=0.0, coverage_lambda=0.0,
            mask_mode="dense", gamma=DEFAULT_GAMMA,
            gamma_mode=DEFAULT_GAMMA_MODE, gamma_floor=DEFAULT_GAMMA_FLOOR,
            norm_guard=DEFAULT_NORM_GUARD,
            consensus_alpha=DEFAULT_CONSENSUS_ALPHA,
            alignment_alpha=DEFAULT_ALIGNMENT_ALPHA):
    """Full LASRC pipeline: masks -> residual compose -> coverage reg.

    Returns
    -------
    final_state_dict   : dict
    coverage_reg_value : float
    layer_stats        : list[dict]
    """
    layer_groups = precomputed["layer_groups"]
    layer_budgets = precomputed["layer_budgets"]
    expert_norms = precomputed["expert_norms"]

    masks = build_layer_masks(
        weights,
        expert_norms,
        layer_budgets,
        layer_groups,
        mask_mode=mask_mode,
    )
    sd, stats = compose_state_dict(
        cache, lora_module_list, weights,
        precomputed, masks,
        prune_threshold=prune_threshold,
        gamma=gamma,
        gamma_mode=gamma_mode,
        gamma_floor=gamma_floor,
        norm_guard=norm_guard,
        consensus_alpha=consensus_alpha,
        alignment_alpha=alignment_alpha,
    )
    cov = coverage_regularization(masks, lam=coverage_lambda)
    return sd, cov, stats
