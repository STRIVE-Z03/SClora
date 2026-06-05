from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
if not hasattr(numpy, "float_"):
    numpy.float_ = numpy.float64
if not hasattr(numpy, "int_"):
    numpy.int_ = numpy.int64
import random
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig
from functools import partial
from typing import List, Optional, Union
import copy


def _import_nevergrad():
    try:
        import nevergrad as ng
    except ImportError as exc:
        raise ImportError(
            "nevergrad and its compatible bayesian-optimization dependency are required "
            "for LoRA composition. Install `nevergrad==1.0.2` and "
            "`bayesian-optimization<2`."
        ) from exc
    return ng

def set_model_weights(model, weights, cache):
    final_state_dict = {}
    keys = cache[list(cache.keys())[0]].keys()
    for i, peft_model_id in enumerate(cache.keys()):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    set_peft_model_state_dict(model, final_state_dict)
    # model = model.merge_and_unload()
    return model

def load_base_model_and_lora_modules(lora_module_list=None, model_name_or_path: Optional[str] = None):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'lora_list:{lora_module_list}')
    if lora_module_list is None or len(lora_module_list) == 0:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer, None, None
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
        
    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
    return peft_model, tokenizer, cache, base_model


def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(example_dataset, model, batch_size):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    data_batch_size = len(example_dataset) if batch_size is None else min(len(example_dataset), batch_size)
    # use gpu if available
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset["input"])

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares



def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular,
              compose_mode='additive', lasrc_precomputed=None, lasrc_config=None):
    if compose_mode == 'lasrc':
        import lasrc
        cfg = lasrc_config or {}
        final_state_dict, cov_reg, _ = lasrc.compose(
            cache, list(cache.keys()), weights,
            precomputed=lasrc_precomputed,
            prune_threshold=cfg.get('prune_threshold', 0.0),
            coverage_lambda=cfg.get('coverage_lambda', 0.0),
            mask_mode=cfg.get('mask_mode', 'dense'),
            gamma=cfg.get('gamma', lasrc.DEFAULT_GAMMA),
            gamma_mode=cfg.get('gamma_mode', lasrc.DEFAULT_GAMMA_MODE),
            gamma_floor=cfg.get('gamma_floor', lasrc.DEFAULT_GAMMA_FLOOR),
            norm_guard=cfg.get('norm_guard', lasrc.DEFAULT_NORM_GUARD),
            consensus_alpha=cfg.get('consensus_alpha', lasrc.DEFAULT_CONSENSUS_ALPHA),
            alignment_alpha=cfg.get('alignment_alpha', lasrc.DEFAULT_ALIGNMENT_ALPHA),
        )
        set_peft_model_state_dict(model, final_state_dict)
        loss = get_loss(example_dataset, model, batch_size)
        metric_val = loss + get_regular(weights) + cov_reg
        return metric_val

    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val


def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict
    
def lorahub_inference(example_inputs: List[str],
                      model_or_name_path: Union[AutoModelForSeq2SeqLM, str],
                      tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
                      batch_size: int,
                      # if not provided, we do not report the accuracy
                      example_outputs: List[str]=None,detail=False):
    
    def accuracy_score(outputs, ground_truths, detail=False):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                correct += 1
            total += 1
        if detail:
            return correct / total * 100, correct, total
        return correct / total * 100

    example_predictions = []
    # load model
    if isinstance(model_or_name_path, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_or_name_path)
    else:
        model = model_or_name_path
    
    # load tokenizer
    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path)
    else:
        tokenizer = tokenizer_or_tokenizer_path
            
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=256
        )
        outputs = tokenizer.batch_decode(
            outputs.to("cpu"), skip_special_tokens=True
        )
        example_predictions.extend(outputs)
    
    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs, detail)
    else:
        task_perf = None
        
    return example_predictions, task_perf


def lorahub_learning(model,
                     cache,
                     tokenizer,
                     lora_module_list: List[str], 
                     example_inputs: List[str], 
                     example_outputs: List[str], 
                     max_inference_step: int,
                     batch_size=None,
                     get_loss=default_get_loss, 
                     get_regular=default_l1_regularization,
                     seed=42,
                     mode='',
                     start_weights=0.0,
                     layer_to_change = [0,24],
                     compose_mode='additive',
                     lasrc_config=None,
                     return_diagnostics=False):
    # set seed for reproducibility
    ng = _import_nevergrad()
    random.seed(seed)
    numpy.random.seed(seed)
    
    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 
    keys = cache[lora_module_list[0]].keys()
    assert layer_to_change[1] - layer_to_change[0] >= 0, "The layer_to_change should be a valid range"
    number_of_layers = layer_to_change[1] - layer_to_change[0]
    number_of_keys = len(keys)

    # LASRC: pre-compute overlap / budgets / norms ONCE (weight-independent)
    lasrc_pre = None
    if compose_mode == 'lasrc':
        import lasrc
        cfg = lasrc_config or {}
        lasrc_pre = lasrc.precompute(
            cache, lora_module_list,
            k_min=cfg.get('k_min', 4),
            k_max=cfg.get('k_max', 12),
            beta=cfg.get('beta', 0.5),
        )
        print("> LASRC pre-computation done: %d layer groups, budgets=%s, overlap_range=(%.4f, %.4f), mask_mode=%s, gamma=%.3f, gamma_mode=%s, consensus_alpha=%.3f, alignment_alpha=%.3f" % (
            len(lasrc_pre['layer_budgets']),
            dict(sorted(lasrc_pre['layer_budgets'].items())),
            lasrc_pre.get('min_mean_overlap', 0.0),
            lasrc_pre.get('max_mean_overlap', 0.0),
            cfg.get('mask_mode', 'dense'),
            cfg.get('gamma', lasrc.DEFAULT_GAMMA),
            cfg.get('gamma_mode', lasrc.DEFAULT_GAMMA_MODE),
            cfg.get('consensus_alpha', lasrc.DEFAULT_CONSENSUS_ALPHA),
            cfg.get('alignment_alpha', lasrc.DEFAULT_ALIGNMENT_ALPHA),
        ))
   
    get_score_partial = partial(get_score, 
                            model=model, 
                            cache=cache,
                            example_dataset=dataset,
                            batch_size=batch_size,
                            get_loss=get_loss, 
                            get_regular=get_regular,
                            compose_mode=compose_mode,
                            lasrc_precomputed=lasrc_pre,
                            lasrc_config=lasrc_config)
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[start_weights] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    if mode == 'succeed':
        instrum = ng.p.Array(
            init=start_weights,
            upper=[1.5] * len(start_weights),
            lower=[-1.5] * len(start_weights),
        )

    if number_of_layers > 0:
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
        print("> Begin to perform gradient-free optimization ...")
        recommendation = optimizer.minimize(get_score_partial, verbosity=0)
    
    # Apply final weights using the appropriate composition mode
    if compose_mode == 'lasrc':
        import lasrc
        cfg = lasrc_config or {}
        final_sd, _, final_stats = lasrc.compose(
            cache, lora_module_list, recommendation.value,
            precomputed=lasrc_pre,
            prune_threshold=cfg.get('prune_threshold', 0.0),
            coverage_lambda=cfg.get('coverage_lambda', 0.0),
            mask_mode=cfg.get('mask_mode', 'dense'),
            gamma=cfg.get('gamma', lasrc.DEFAULT_GAMMA),
            gamma_mode=cfg.get('gamma_mode', lasrc.DEFAULT_GAMMA_MODE),
            gamma_floor=cfg.get('gamma_floor', lasrc.DEFAULT_GAMMA_FLOOR),
            norm_guard=cfg.get('norm_guard', lasrc.DEFAULT_NORM_GUARD),
            consensus_alpha=cfg.get('consensus_alpha', lasrc.DEFAULT_CONSENSUS_ALPHA),
            alignment_alpha=cfg.get('alignment_alpha', lasrc.DEFAULT_ALIGNMENT_ALPHA),
        )
        if final_stats:
            gammas = [float(item.get('gamma', 0.0)) for item in final_stats]
            budgets = [int(item.get('budget', 0)) for item in final_stats]
            alignments = [float(item.get('alignment', 1.0)) for item in final_stats]
            ratios = []
            for item in final_stats:
                linear_norm = float(item.get('linear_norm', 0.0))
                residual_norm = float(item.get('residual_norm', 0.0))
                if linear_norm > 1e-9:
                    ratios.append(residual_norm / linear_norm)
            if ratios:
                print("> LASRC final stats: gamma[min/mean/max]=%.4f/%.4f/%.4f budget[min/mean/max]=%d/%.2f/%d residual_ratio[min/mean/max]=%.4f/%.4f/%.4f alignment[min/mean/max]=%.4f/%.4f/%.4f" % (
                    min(gammas),
                    sum(gammas) / len(gammas),
                    max(gammas),
                    min(budgets),
                    sum(budgets) / len(budgets),
                    max(budgets),
                    min(ratios),
                    sum(ratios) / len(ratios),
                    max(ratios),
                    min(alignments),
                    sum(alignments) / len(alignments),
                    max(alignments),
                ))
        set_peft_model_state_dict(model, final_sd)
    else:
        final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
        set_peft_model_state_dict(model, final_lora)
    # model = model.merge_and_unload()
    if return_diagnostics:
        return recommendation.value, model, tokenizer, {}
    return recommendation.value, model, tokenizer
