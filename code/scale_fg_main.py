"""
SCALE-FG: finite-grid support-time adapter-state selection.

For each task, this script builds a small set of candidate LoRA adapter
states and selects the candidate with the lowest support loss before query
evaluation. Candidate states use either linear composition or LASRC.
"""

import argparse
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import algorithm
import lasrc
import utils


DEFAULT_VIEWS = [
    "flan_v2_emb_retrieval_focused.json",
    "flan_v2_emb_balanced.json",
    "flan_v2_emb_qa_only.json",
]
DEFAULT_SPARSE_RATES = [0.0, 0.3]
DEFAULT_OPS = ["linear", "lasrc"]
PRIMARY_VIEW_INDEX = 0
DEFAULT_SPARSE_REFERENCE_RATE = 0.3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SCALE-FG primitive candidate construction.",
    )
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-large")
    parser.add_argument(
        "--embedding_model_name_or_path",
        type=str,
        default="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
    )
    parser.add_argument("--seed", type=int, default=133)
    utils.add_few_shot_args(parser)

    parser.add_argument("--global_top_k", type=int, default=20)
    parser.add_argument("--optimization_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=10)
    parser.add_argument(
        "--retrieval_metric",
        choices=["cosine", "l2", "bm25"],
        default="cosine",
    )

    parser.add_argument(
        "--views",
        type=str,
        default=",".join(DEFAULT_VIEWS),
        help="Comma-separated LoRA-index filenames under xx/xx.",
    )
    parser.add_argument(
        "--sparse_rates",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SPARSE_RATES),
        help="Comma-separated sparse candidate rates. Use 0 for the dense candidate.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=",".join(DEFAULT_OPS),
        help="Comma-separated composition operators: linear, lasrc.",
    )
    parser.add_argument(
        "--primary_view",
        type=str,
        default="",
        help="Optional view filename used for the LASRC reference row.",
    )
    parser.add_argument(
        "--sparse_reference_rate",
        type=float,
        default=DEFAULT_SPARSE_REFERENCE_RATE,
        help="Sparse rate reported as the SCALE-FG sparse reference.",
    )

    parser.add_argument("--lasrc_k_min", type=int, default=4)
    parser.add_argument("--lasrc_k_max", type=int, default=12)
    parser.add_argument("--lasrc_beta", type=float, default=0.5)
    parser.add_argument("--lasrc_prune_threshold", type=float, default=0.0)
    parser.add_argument("--lasrc_coverage_lambda", type=float, default=0.0)
    parser.add_argument("--lasrc_mask_mode", choices=["dense", "scheduled"], default="dense")
    parser.add_argument("--lasrc_gamma", type=float, default=lasrc.DEFAULT_GAMMA)
    parser.add_argument(
        "--lasrc_gamma_mode",
        choices=["fixed", "overlap"],
        default=lasrc.DEFAULT_GAMMA_MODE,
    )
    parser.add_argument("--lasrc_gamma_floor", type=float, default=lasrc.DEFAULT_GAMMA_FLOOR)
    parser.add_argument("--lasrc_norm_guard", type=float, default=lasrc.DEFAULT_NORM_GUARD)
    parser.add_argument(
        "--lasrc_consensus_alpha",
        type=float,
        default=lasrc.DEFAULT_CONSENSUS_ALPHA,
    )
    parser.add_argument(
        "--lasrc_alignment_alpha",
        type=float,
        default=lasrc.DEFAULT_ALIGNMENT_ALPHA,
    )

    parser.add_argument("--result_tag", type=str, default="scale_fg")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to xx/xx/results/scale_fg/<result_tag>.",
    )
    parser.add_argument("--task_subset", type=str, default="")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--max_queries_per_task", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--task_jsonl", type=str, default="")
    return parser.parse_args()


def parse_csv_list(raw_value, caster=str):
    items = []
    for part in raw_value.split(","):
        token = part.strip()
        if token:
            items.append(caster(token))
    return items


def prompt_text(question, answer=None):
    if answer is None:
        return f"Question: {question}"
    return f"Question: {question}\nAnswer: {answer}"


def tokenize_text(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def canonical_answer(text):
    return str(text).strip().lower().replace(".", "")


def is_exact_match(prediction, target):
    return canonical_answer(prediction) == canonical_answer(target)


class SimpleBM25:
    def __init__(self, documents):
        self.documents = [tokenize_text(document) for document in documents]
        self.doc_lengths = [len(document) for document in self.documents]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freq = Counter()
        for document in self.documents:
            for token in set(document):
                self.doc_freq[token] += 1
        self.doc_term_freqs = [Counter(document) for document in self.documents]
        self.size = len(self.documents)
        self.k1 = 1.5
        self.b = 0.75

    def score(self, query_text):
        query_terms = tokenize_text(query_text)
        if not query_terms or not self.documents:
            return [0.0] * self.size

        scores = []
        for term_freqs, doc_length in zip(self.doc_term_freqs, self.doc_lengths):
            score = 0.0
            for token in query_terms:
                token_freq = term_freqs.get(token, 0)
                if token_freq == 0:
                    continue
                doc_count = self.doc_freq.get(token, 0)
                idf = math.log(((self.size - doc_count + 0.5) / (doc_count + 0.5)) + 1.0)
                denom = token_freq + self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(self.avgdl, 1e-12)
                )
                score += idf * (token_freq * (self.k1 + 1.0)) / max(denom, 1e-12)
            scores.append(score)
        return scores


def _to_query_tensor(query_embedding):
    if isinstance(query_embedding, torch.Tensor):
        return query_embedding.to(dtype=torch.float32)
    return torch.tensor(query_embedding, dtype=torch.float32)


def rank_indices_with_scores(
    metric,
    top_k,
    candidate_embedding=None,
    query_embedding=None,
    bm25_index=None,
    query_text=None,
):
    if metric == "cosine":
        query_vector = _to_query_tensor(query_embedding)
        scores = torch.cosine_similarity(candidate_embedding, query_vector, dim=1)
    elif metric == "l2":
        query_vector = _to_query_tensor(query_embedding)
        scores = -torch.norm(candidate_embedding - query_vector, dim=1)
    elif metric == "bm25":
        scores = torch.tensor(bm25_index.score(query_text), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported retrieval metric: {metric}")

    ranked = torch.argsort(scores, descending=True)
    chosen = ranked[:top_k].tolist()
    chosen_scores = [float(scores[index]) for index in chosen]
    return chosen, chosen_scores


def select_top_loras(
    candidate_lora_path,
    top_k,
    metric,
    candidate_lora_embedding=None,
    centroid=None,
    bm25_index=None,
    query_text=None,
):
    indices, scores = rank_indices_with_scores(
        metric=metric,
        top_k=top_k,
        candidate_embedding=candidate_lora_embedding,
        query_embedding=centroid,
        bm25_index=bm25_index,
        query_text=query_text,
    )
    return [candidate_lora_path[index] for index in indices], scores


def sparse_candidate_cache(cache, lora_module_list, sparse_rate, seed):
    """Create a SCALE-FG sparse candidate by masking LoRA delta tensors."""
    if sparse_rate <= 0.0:
        return cache
    if sparse_rate >= 1.0:
        raise ValueError("sparse_rate must be < 1.0, got %.3f" % sparse_rate)

    new_cache = {}
    rng = torch.Generator()
    rescale_factor = 1.0 / (1.0 - sparse_rate)
    for idx, module_id in enumerate(lora_module_list):
        if module_id not in cache:
            continue
        new_state = {}
        module_seed = int(seed) + idx
        for key, tensor in cache[module_id].items():
            key_hash = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            rng.manual_seed(module_seed + key_hash)
            keep_probability = torch.full(tensor.shape, 1.0 - sparse_rate, dtype=torch.float32)
            mask = torch.bernoulli(keep_probability, generator=rng)
            mask = mask.to(dtype=tensor.dtype, device=tensor.device)
            sparse_tensor = tensor * mask * rescale_factor
            original_norm = tensor.float().norm().item()
            sparse_norm = sparse_tensor.float().norm().item()
            if original_norm > 1e-12 and sparse_norm > 1e-12:
                sparse_tensor = sparse_tensor * (original_norm / sparse_norm)
            new_state[key] = sparse_tensor
        new_cache[module_id] = new_state

    for module_id in cache:
        if module_id not in new_cache:
            new_cache[module_id] = cache[module_id]
    return new_cache


def grid_point_id(view, sparse_rate, op):
    return "view={view}|sparse={sparse}|op={op}".format(
        view=Path(view).name,
        sparse=str(sparse_rate),
        op=op,
    )


def build_primitive_grid(views, sparse_rates, ops):
    grid = []
    for view in views:
        for rate in sparse_rates:
            for op in ops:
                grid.append(
                    {
                        "id": grid_point_id(view, rate, op),
                        "view": Path(view).name,
                        "view_path": view,
                        "sparse_rate": float(rate),
                        "op": op,
                    }
                )
    return grid


def load_view(view_filename):
    lora_message = utils.load_lora_index(view_filename, require_local=True)
    candidate_paths = list(lora_message.keys())
    candidate_embedding = torch.tensor(
        [lora_message[key]["centroid"] for key in lora_message],
        dtype=torch.float32,
    )
    candidate_texts = [lora_message[key].get("text", key) for key in lora_message]
    return {
        "candidate_paths": candidate_paths,
        "candidate_embedding": candidate_embedding,
        "candidate_bm25": SimpleBM25(candidate_texts),
    }


def shortlist_for_view(view_state, example_inputs, example_outputs, embedding_model, args):
    example_texts = [
        prompt_text(question, answer)
        for question, answer in zip(example_inputs, example_outputs)
    ]
    example_embeddings = embedding_model.encode(example_texts)
    centroid = torch.tensor(example_embeddings, dtype=torch.float32).mean(dim=0)
    query_text = "\n".join(example_texts)
    shortlist, scores = select_top_loras(
        candidate_lora_path=view_state["candidate_paths"],
        top_k=int(args.global_top_k),
        metric=args.retrieval_metric,
        candidate_lora_embedding=view_state["candidate_embedding"],
        centroid=centroid,
        bm25_index=view_state["candidate_bm25"],
        query_text=query_text,
    )
    return utils.normalize_lora_identifiers(shortlist), [float(score) for score in scores]


def make_lasrc_config(args):
    return {
        "k_min": int(args.lasrc_k_min),
        "k_max": int(args.lasrc_k_max),
        "beta": float(args.lasrc_beta),
        "prune_threshold": float(args.lasrc_prune_threshold),
        "coverage_lambda": float(args.lasrc_coverage_lambda),
        "mask_mode": args.lasrc_mask_mode,
        "gamma": float(args.lasrc_gamma),
        "gamma_mode": args.lasrc_gamma_mode,
        "gamma_floor": float(args.lasrc_gamma_floor),
        "norm_guard": float(args.lasrc_norm_guard),
        "consensus_alpha": float(args.lasrc_consensus_alpha),
        "alignment_alpha": float(args.lasrc_alignment_alpha),
    }


def measure_support(model, tokenizer, example_inputs, example_outputs, args):
    support_dataset = algorithm.load_dataset(example_inputs, example_outputs, tokenizer)
    support_loss = algorithm.default_get_loss(support_dataset, model, int(args.batch_size))
    _, support_perf = algorithm.lorahub_inference(
        example_inputs=example_inputs,
        model_or_name_path=model,
        tokenizer_or_tokenizer_path=tokenizer,
        batch_size=int(args.inference_batch_size),
        example_outputs=example_outputs,
        detail=True,
    )
    _, correct, total = support_perf
    support_accuracy = correct / max(total, 1) * 100.0
    return float(support_loss), float(support_accuracy)


def decode_queries(model, tokenizer, query_inputs, query_outputs, args):
    if not query_inputs:
        return [], 0.0
    predictions, perf = algorithm.lorahub_inference(
        example_inputs=query_inputs,
        model_or_name_path=model,
        tokenizer_or_tokenizer_path=tokenizer,
        batch_size=int(args.inference_batch_size),
        example_outputs=query_outputs,
        detail=True,
    )
    _, correct, total = perf
    records = []
    for qid, (prediction, target) in enumerate(zip(predictions, query_outputs)):
        records.append(
            {
                "query_id": int(qid),
                "prediction": prediction,
                "target": target,
                "correct": bool(is_exact_match(prediction, target)),
            }
        )
    return records, (correct / total * 100.0 if total else 0.0)


def evaluate_grid_point(shortlist, sparse_rate, op, example_inputs, example_outputs,
                        query_inputs, query_outputs, args, lasrc_config):
    model, tokenizer, raw_cache, _ = algorithm.load_base_model_and_lora_modules(
        shortlist,
        args.model_name_or_path,
    )
    cache = sparse_candidate_cache(raw_cache, shortlist, float(sparse_rate), int(args.seed))
    compose_mode = "lasrc" if op == "lasrc" else "additive"

    start_time = time.time()
    weights, model, tokenizer = algorithm.lorahub_learning(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        lora_module_list=shortlist,
        example_inputs=example_inputs,
        example_outputs=example_outputs,
        max_inference_step=int(args.optimization_steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        mode="",
        start_weights=0.0,
        compose_mode=compose_mode,
        lasrc_config=lasrc_config if compose_mode == "lasrc" else None,
    )
    support_loss, support_accuracy = measure_support(
        model, tokenizer, example_inputs, example_outputs, args
    )
    query_records, query_accuracy = decode_queries(
        model, tokenizer, query_inputs, query_outputs, args
    )
    elapsed = time.time() - start_time

    del model, tokenizer, raw_cache, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "support_loss": float(support_loss),
        "support_accuracy": float(support_accuracy),
        "query_accuracy": float(query_accuracy),
        "query_correct": int(sum(1 for record in query_records if record["correct"])),
        "query_total": int(len(query_records)),
        "query_predictions": query_records,
        "shortlist": [Path(path).name for path in shortlist],
        "optimized_weights": [float(weight) for weight in weights],
        "wall_seconds": float(elapsed),
    }


def run_task(task_name, question_path, args, embedding_model,
             view_cache, primitive_grid, primary_view, lasrc_config):
    with question_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    inputs = [example["input"] for example in data["examples"]]
    outputs = [example["target"] for example in data["examples"]]
    source_indices = list(range(len(inputs)))
    selection_seed = args.seed if args.selection_seed is None else args.selection_seed

    if args.disable_shuffle:
        ordered_inputs = list(inputs)
        ordered_outputs = list(outputs)
        ordered_source = list(source_indices)
    else:
        ordered_inputs, ordered_outputs, ordered_source = utils.shuffle_list_with_seed(
            inputs,
            outputs,
            source_indices,
            seed=selection_seed,
        )

    (
        example_inputs,
        example_outputs,
        query_inputs,
        query_outputs,
        selected_indices,
    ) = utils.prepare_few_shot_split(
        ordered_inputs,
        ordered_outputs,
        example_num=int(args.example_num),
        example_selection=args.example_selection,
        example_indices=args.example_indices,
        example_offset=int(args.example_offset),
        print_selected_examples=False,
        task_name=task_name,
    )

    if args.max_queries_per_task and args.max_queries_per_task > 0:
        query_inputs = query_inputs[: int(args.max_queries_per_task)]
        query_outputs = query_outputs[: int(args.max_queries_per_task)]

    view_shortlists = {}
    for view_path in {point["view_path"] for point in primitive_grid}:
        shortlist, scores = shortlist_for_view(
            view_cache[view_path],
            example_inputs,
            example_outputs,
            embedding_model,
            args,
        )
        view_shortlists[view_path] = {"shortlist": shortlist, "scores": scores}

    points = []
    for idx, point in enumerate(primitive_grid):
        print(
            "[%s] grid %d/%d view=%s sparse=%s op=%s"
            % (task_name, idx + 1, len(primitive_grid), point["view"], point["sparse_rate"], point["op"]),
            flush=True,
        )
        shortlist = view_shortlists[point["view_path"]]["shortlist"]
        evaluation = evaluate_grid_point(
            shortlist=shortlist,
            sparse_rate=point["sparse_rate"],
            op=point["op"],
            example_inputs=example_inputs,
            example_outputs=example_outputs,
            query_inputs=query_inputs,
            query_outputs=query_outputs,
            args=args,
            lasrc_config=lasrc_config,
        )
        points.append({**point, **evaluation})
        print(
            "  support_loss=%.4f support_acc=%.2f query_acc=%.2f"
            % (evaluation["support_loss"], evaluation["support_accuracy"], evaluation["query_accuracy"]),
            flush=True,
        )

    return {
        "task_name": task_name,
        "question_path": "xx/xx",
        "primary_view": Path(primary_view).name,
        "sparse_reference_rate": float(args.sparse_reference_rate),
        "few_shot": {
            "selected_indices": [int(index) for index in selected_indices],
            "selected_dataset_indices": [int(ordered_source[index]) for index in selected_indices],
            "selection_seed": int(selection_seed),
            "example_num": int(args.example_num),
            "example_selection": args.example_selection,
            "example_offset": int(args.example_offset),
            "disable_shuffle": bool(args.disable_shuffle),
        },
        "view_shortlists": {
            Path(view_path).name: {
                "shortlist": [Path(path).name for path in payload["shortlist"]],
                "scores": payload["scores"],
            }
            for view_path, payload in view_shortlists.items()
        },
        "query_total": int(len(query_inputs)),
        "primitive_grid": points,
    }


def select_tasks(args):
    files = utils.question_files()
    if args.task_subset:
        wanted = {part.strip() for part in args.task_subset.split(",") if part.strip()}
        files = [path for path in files if path.stem in wanted]
    if args.max_tasks and args.max_tasks > 0:
        files = files[: int(args.max_tasks)]
    return files


def main():
    args = parse_args()
    utils.seed_everything(int(args.seed))
    random.seed(int(args.seed))

    views = parse_csv_list(args.views, str)
    sparse_rates = [float(value) for value in parse_csv_list(args.sparse_rates, str)]
    ops = parse_csv_list(args.ops, str)
    for op in ops:
        if op not in {"linear", "lasrc"}:
            raise ValueError("Unsupported composition op: %r. Use linear or lasrc." % op)
    for view in views:
        utils.load_lora_index(view, require_local=True)

    primary_view = args.primary_view or views[PRIMARY_VIEW_INDEX]
    if primary_view not in views:
        raise ValueError("--primary_view %r is not in --views %r" % (primary_view, views))

    primitive_grid = build_primitive_grid(views, sparse_rates, ops)
    lasrc_config = make_lasrc_config(args)

    print("============================================================")
    print("SCALE-FG finite-grid adapter-state selection")
    print("model        : %s" % args.model_name_or_path)
    print("embedder     : %s" % args.embedding_model_name_or_path)
    print("views        : %s" % ", ".join(Path(view).name for view in views))
    print("sparse rates : %s" % ", ".join(str(rate) for rate in sparse_rates))
    print("ops          : %s" % ", ".join(ops))
    print("primary view : %s" % Path(primary_view).name)
    print("grid size    : %d primitive points / task" % len(primitive_grid))
    print("============================================================")

    embedding_model = SentenceTransformer(args.embedding_model_name_or_path)
    view_cache = {view: load_view(view) for view in views}
    tasks = select_tasks(args)
    if not tasks:
        raise FileNotFoundError("No task JSON files found under xx/xx/question_list")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else utils.resolve_resource("results", "scale_fg", args.result_tag)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_writer = None
    if args.task_jsonl:
        trace_writer = utils.ensure_parent(args.task_jsonl).open("a", encoding="utf-8")

    run_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": args.model_name_or_path,
        "embedding_model_name_or_path": args.embedding_model_name_or_path,
        "resource_root": "xx/xx",
        "views": [Path(view).name for view in views],
        "sparse_rates": [float(rate) for rate in sparse_rates],
        "ops": list(ops),
        "primary_view": Path(primary_view).name,
        "sparse_reference_rate": float(args.sparse_reference_rate),
        "global_top_k": int(args.global_top_k),
        "optimization_steps": int(args.optimization_steps),
        "batch_size": int(args.batch_size),
        "inference_batch_size": int(args.inference_batch_size),
        "retrieval_metric": args.retrieval_metric,
        "seed": int(args.seed),
        "selection_seed": None if args.selection_seed is None else int(args.selection_seed),
        "example_num": int(args.example_num),
        "example_selection": args.example_selection,
        "example_offset": int(args.example_offset),
        "disable_shuffle": bool(args.disable_shuffle),
        "lasrc_config": lasrc_config,
        "result_tag": args.result_tag,
        "output_dir": "xx/xx",
        "task_subset": args.task_subset,
        "max_tasks": int(args.max_tasks),
        "max_queries_per_task": int(args.max_queries_per_task),
        "primitive_grid_size": len(primitive_grid),
    }
    utils.save_json(output_dir / "_run_meta.json", run_meta)

    all_task_payloads = []
    for task_index, question_path in enumerate(tasks):
        task_name = question_path.stem
        per_task_path = output_dir / ("task_%s.json" % task_name)
        if args.skip_existing and per_task_path.exists():
            print("[%d/%d] %s : skip existing" % (task_index + 1, len(tasks), task_name))
            with per_task_path.open("r", encoding="utf-8") as file:
                all_task_payloads.append(json.load(file))
            continue

        print("[%d/%d] %s : evaluating" % (task_index + 1, len(tasks), task_name))
        start_time = time.time()
        payload = run_task(
            task_name=task_name,
            question_path=question_path,
            args=args,
            embedding_model=embedding_model,
            view_cache=view_cache,
            primitive_grid=primitive_grid,
            primary_view=primary_view,
            lasrc_config=lasrc_config,
        )
        payload["wall_seconds_total"] = float(time.time() - start_time)
        utils.save_json(per_task_path, payload)
        if trace_writer is not None:
            trace_writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            trace_writer.flush()
        all_task_payloads.append(payload)

    if trace_writer is not None:
        trace_writer.close()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_tag": args.result_tag,
        "tasks": sorted(payload["task_name"] for payload in all_task_payloads),
        "num_tasks": len(all_task_payloads),
        "primitive_grid_size": len(primitive_grid),
        "run_meta_path": "xx/xx/_run_meta.json",
    }
    utils.save_json(output_dir / "_summary.json", summary)
    if not all_task_payloads:
        raise RuntimeError("SCALE-FG produced zero valid task JSON files.")

    print("============================================================")
    print("SCALE-FG primitive evaluation done.")
    print("tasks evaluated : %d / %d" % (len(all_task_payloads), len(tasks)))
    print("output          : %s" % output_dir)
    print("next step       : python code/scale_fg_table.py --input_dir %s" % output_dir)
    print("============================================================")


if __name__ == "__main__":
    main()
