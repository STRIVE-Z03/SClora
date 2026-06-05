import json
import os
import random
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = Path(
    os.environ.get("SCALE_RESOURCE_ROOT")
    or os.environ.get("RESOURCE_ROOT")
    or "xx/xx"
).expanduser()


def resolve_resource(*parts):
    return RESOURCE_ROOT.joinpath(*parts)


def resolve_output_path(path):
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return REPO_ROOT / output_path


def ensure_parent(path):
    output_path = resolve_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(path, payload):
    output_path = ensure_parent(path)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def question_files():
    question_root = resolve_resource("question_list")
    return sorted(question_root.glob("*.json"))


def is_complete_lora_dir(path):
    if not path.is_dir():
        return False
    if not (path / "adapter_config.json").exists():
        return False
    return any(
        (path / filename).exists()
        for filename in ("adapter_model.bin", "adapter_model.safetensors")
    )


def local_lora_dirs():
    lora_root = resolve_resource("lora_list")
    if not lora_root.exists():
        return []
    return sorted(path for path in lora_root.iterdir() if is_complete_lora_dir(path))


def local_lora_dir_names():
    return {path.name for path in local_lora_dirs()}


def normalize_lora_identifier(identifier):
    identifier_path = Path(identifier)
    if identifier_path.exists():
        return str(identifier_path)

    local_path = resolve_resource("lora_list", identifier)
    if local_path.exists():
        return str(local_path)

    return identifier


def normalize_lora_identifiers(identifiers):
    return [normalize_lora_identifier(identifier) for identifier in identifiers]


def ordered_unique(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def missing_local_lora_identifiers(identifiers):
    available_names = local_lora_dir_names()
    missing = []
    for identifier in identifiers:
        identifier_path = Path(identifier)
        if is_complete_lora_dir(identifier_path):
            continue
        if Path(str(identifier)).name in available_names:
            continue
        missing.append(identifier)
    return missing


def require_local_lora_resources(identifiers=None, expected_count=97):
    lora_root = resolve_resource("lora_list")
    local_dirs = local_lora_dirs()
    if not lora_root.exists():
        raise FileNotFoundError(f"Missing local LoRA directory: {lora_root}")

    if expected_count is not None and len(local_dirs) != expected_count:
        raise ValueError(
            f"The paper setting expects {expected_count} local LoRA modules under {lora_root}, "
            f"but found {len(local_dirs)}."
        )

    if identifiers is None:
        return local_dirs

    missing = missing_local_lora_identifiers(identifiers)
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        raise FileNotFoundError(
            f"Index references {len(missing)} missing local LoRA modules under {lora_root}. "
            f"Examples: {preview}."
        )

    return local_dirs


def load_lora_index(filename="flan_v2_emb.json", expected_count=97, require_local=False):
    index_path = resolve_resource(filename)
    with index_path.open("r", encoding="utf-8") as file:
        lora_index = json.load(file)

    if expected_count is not None and len(lora_index) != expected_count:
        raise ValueError(
            f"{index_path} should contain {expected_count} LoRA entries for the paper setting, "
            f"but found {len(lora_index)}."
        )

    if require_local:
        require_local_lora_resources(lora_index.keys(), expected_count=expected_count)

    return lora_index


def load_paper97_names(filename="lora_list.paper97_original.txt"):
    path = resolve_resource(filename)
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        names.add(Path(stripped).name)
    return names


def save_results(path, task_accs):
    output_path = ensure_parent(path)
    scores = dict(task_accs)
    scores["average"] = (
        sum(scores.values()) / len(scores) if scores else 0.0
    )
    with output_path.open("w", encoding="utf-8") as file:
        for task_name, value in scores.items():
            print(f"{task_name}:{value}")
            file.write(f"{task_name}:{value}\n")
    return output_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def shuffle_list(*lists):
    shuffled = list(zip(*lists))
    random.shuffle(shuffled)
    if not shuffled:
        return tuple([] for _ in lists)
    return tuple(list(items) for items in zip(*shuffled))


def shuffle_list_with_seed(*lists, seed):
    shuffled = list(zip(*lists))
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return tuple([] for _ in lists)
    return tuple(list(items) for items in zip(*shuffled))


def add_few_shot_args(parser):
    parser.add_argument(
        "--example_num",
        type=int,
        default=5,
        help="Number of few-shot examples to use per task.",
    )
    parser.add_argument(
        "--example_selection",
        choices=["head", "spread", "manual"],
        default="head",
        help=(
            "How to choose few-shot examples from the ordered task data. "
            "Use --disable_shuffle to keep the original dataset order."
        ),
    )
    parser.add_argument(
        "--example_indices",
        type=str,
        default="",
        help="Comma-separated zero-based indices for --example_selection manual.",
    )
    parser.add_argument(
        "--example_offset",
        type=int,
        default=0,
        help="Skip this many ordered examples before selecting few-shot examples.",
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=None,
        help="Seed used only for task-example shuffling. Defaults to the script's main seed.",
    )
    parser.add_argument(
        "--disable_shuffle",
        action="store_true",
        help="Keep task examples in dataset order instead of shuffling before selection.",
    )
    parser.add_argument(
        "--print_selected_examples",
        action="store_true",
        help="Print the chosen few-shot indices and short text previews.",
    )
    return parser


def parse_example_indices(raw_value):
    if not raw_value:
        return []
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def evenly_spaced_indices(indices, count):
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[0]]

    positions = [
        round(step * (len(indices) - 1) / (count - 1))
        for step in range(count)
    ]
    selected = ordered_unique(indices[position] for position in positions)
    if len(selected) < count:
        selected_set = set(selected)
        for index in indices:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) == count:
                break
    return selected


def choose_few_shot_indices(
    total_count,
    example_num,
    selection="head",
    example_indices="",
    example_offset=0,
):
    if example_num < 0:
        raise ValueError(f"example_num must be non-negative, got {example_num}.")
    if example_offset < 0:
        raise ValueError(f"example_offset must be non-negative, got {example_offset}.")
    if total_count <= 0 or example_num == 0:
        return []

    available = list(range(example_offset, total_count))
    if not available:
        raise ValueError(
            f"example_offset={example_offset} is out of range for total_count={total_count}."
        )

    target_count = min(example_num, len(available))
    if selection == "head":
        return available[:target_count]
    if selection == "spread":
        return evenly_spaced_indices(available, target_count)
    if selection == "manual":
        parsed_indices = parse_example_indices(example_indices)
        if len(parsed_indices) != target_count:
            raise ValueError(
                "Manual example selection requires exactly "
                f"{target_count} indices, got {len(parsed_indices)}."
            )
        if len(set(parsed_indices)) != len(parsed_indices):
            raise ValueError("Manual example indices must be unique.")
        invalid = [index for index in parsed_indices if index < 0 or index >= total_count]
        if invalid:
            raise ValueError(
                f"Manual example indices out of range for total_count={total_count}: {invalid}"
            )
        return parsed_indices
    raise ValueError(f"Unknown example_selection: {selection}")


def preview_text(text, limit=96):
    flattened = " ".join(str(text).split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


def prepare_few_shot_split(
    inputs,
    outputs,
    example_num=5,
    example_selection="head",
    example_indices="",
    example_offset=0,
    print_selected_examples=False,
    task_name="",
):
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must have the same length.")

    selected_indices = choose_few_shot_indices(
        total_count=len(inputs),
        example_num=example_num,
        selection=example_selection,
        example_indices=example_indices,
        example_offset=example_offset,
    )
    selected_index_set = set(selected_indices)

    example_inputs = [inputs[index] for index in selected_indices]
    example_outputs = [outputs[index] for index in selected_indices]
    remaining_inputs = [
        value for index, value in enumerate(inputs) if index not in selected_index_set
    ]
    remaining_outputs = [
        value for index, value in enumerate(outputs) if index not in selected_index_set
    ]

    if print_selected_examples:
        label = task_name or "task"
        print(f"[few-shot] {label} indices={selected_indices}")
        for position, index in enumerate(selected_indices):
            print(
                f"  shot{position}: idx={index} "
                f"q={preview_text(inputs[index], 88)} "
                f"a={preview_text(outputs[index], 48)}"
            )

    return example_inputs, example_outputs, remaining_inputs, remaining_outputs, selected_indices


def build_selected_example_records(inputs, outputs, selected_indices, dataset_indices=None):
    records = []
    for support_id, ordered_index in enumerate(selected_indices):
        if dataset_indices is None:
            dataset_index = int(ordered_index)
        else:
            dataset_index = int(dataset_indices[ordered_index])
        records.append(
            {
                "support_id": int(support_id),
                "ordered_index": int(ordered_index),
                "dataset_index": dataset_index,
                "input_text": str(inputs[ordered_index]),
                "target_text": str(outputs[ordered_index]),
            }
        )
    return records


def main():
    pass


if __name__ == "__main__":
    main()
