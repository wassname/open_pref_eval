import itertools
from typing import Callable, List, Optional, Union
import warnings
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .datasets import ds2name, get_default_datasets
from .helpers.mem import clear_mem
from .helpers.peft import is_peft_model, set_adapter
from .scoring import score_ipo
from .trainer import DataCollatorForPreference, concatenated_forward, tokenize_dataset


def extract_logps(
    model: PreTrainedModel, 
    batch: dict, 
    step: int, 
    score_fn: Callable = score_ipo, 
    include_raw: bool = False
) -> dict:
    """Extract log probabilities and compute preference scores from model output.
    
    Args:
        model: The language model to evaluate
        batch: Batch of tokenized preference pairs
        step: Current step number for indexing
        score_fn: Scoring function(s) to apply - can be single function or dict of functions
        include_raw: Whether to include raw tensors in output
        
    Returns:
        Dict containing computed scores, probabilities, and debug metrics
    """
    batch_size = batch["chosen_input_ids"].shape[0]
    batch_indices = batch_size * step + torch.arange(batch_size)
    
    model.eval()
    
    # Run forward pass through concatenated model
    with torch.no_grad():
        forward_output = concatenated_forward(model, batch)

    # Extract model outputs and convert to float for numerical stability
    chosen_t_logps = forward_output["chosen_logps"].float()
    rejected_t_logps = forward_output["rejected_logps"].float()
    chosen_mask = forward_output["chosen_mask"].float()
    rejected_mask = forward_output["rejected_mask"].float()

    # Validate that we have valid completions
    if (chosen_mask.sum(1) == 0).any() or (rejected_mask.sum(1) == 0).any():
        import warnings
        warnings.warn(f"Some samples have completions completely masked out. Check the dataset.")

    # Compute preference scores using provided scoring function(s)
    outputs = {}
    if isinstance(score_fn, dict):
        # Multiple scoring functions provided
        for score_name, scoring_func in score_fn.items():
            score_results = scoring_func(
                log_prob_chosen=chosen_t_logps,
                log_prob_rejected=rejected_t_logps,
                mask_chosen=chosen_mask,
                mask_rejected=rejected_mask,
            )
            # Prefix score outputs with score name
            prefixed_scores = {f"score_{score_name}__{key}": value for key, value in score_results.items()}
            
            # Validate scores are finite
            for score_key, score_value in prefixed_scores.items():
                # assert torch.isfinite(score_value).all(), f"Score {score_key} is not finite: {score_value}"
                if torch.isnan(score_value).any():
                    logger.warning(f"Score {score_key} contains NaN values. Check the scoring function.")
            
            outputs.update(prefixed_scores)
        
        # Use the last score's sigmoid as the main probability
        outputs["prob"] = outputs[f"score_{score_name}__sigmoid"]
    else:
        # Single scoring function
        score_results = score_fn(
            log_prob_chosen=chosen_t_logps,
            log_prob_rejected=rejected_t_logps,
            mask_chosen=chosen_mask,
            mask_rejected=rejected_mask,
        )
        prefixed_scores = {f"score__{key}": value for key, value in score_results.items()}
        outputs.update(prefixed_scores)
        outputs["prob"] = outputs["score__sigmoid"]

    # Calculate sequence-level log probabilities and perplexities
    chosen_logp = (chosen_t_logps * chosen_mask).sum(1)
    rejected_logp = (rejected_t_logps * rejected_mask).sum(1)

    # Calculate perplexity for each completion
    chosen_ppl = torch.exp(-chosen_logp / chosen_mask.sum(1))
    rejected_ppl = torch.exp(-rejected_logp / rejected_mask.sum(1))

    # Add all computed metrics to outputs (preserving exact variable names for compatibility)
    outputs.update(
        # Debug: log probabilities and perplexity of completions, can be used to check if coherency is maintained
        _chosen_logps=chosen_logp,
        _rejected_logps=rejected_logp,
        _chosen_ppl=chosen_ppl,
        _rejected_ppl=rejected_ppl,
        # Debug: completion length, for checking if the model is biased
        _l_chosen=(batch["chosen_input_ids"] > 0).sum(-1),
        _l_rejected=(batch["rejected_input_ids"] > 0).sum(-1),
    )
    
    if include_raw:
        outputs.update(
            __chosen_logps=chosen_t_logps,
            __rejected_logps=rejected_t_logps,
            __chosen_logits=forward_output["mean_chosen_logits"],
            __rejected_logits=forward_output["mean_rejected_logits"],
            __chosen_mask=chosen_mask,
            __rejected_mask=rejected_mask,
        )
        clear_mem()

    # Convert all tensors to numpy for compatibility
    outputs = {k: v.detach().float().cpu().numpy() for k, v in outputs.items()}
    
    # Add metadata
    outputs["ds_i"] = batch_indices.cpu().numpy()
    return outputs


@torch.no_grad()
def eval_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Union[Dataset, str],
    adapter_names: Optional[List[str]] = None,
    max_prompt_length: int = 512,
    max_length: Optional[int] = 1024,
    batch_size: int = 2,
    verbose: int = 1,
    num_workers=0,
    **kwargs,
) -> pd.DataFrame:
    """Evaluate model on a preference dataset.
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Dataset to evaluate on, or string in format "dataset_name#split"
        adapter_names: List of adapter names to evaluate (for PEFT models)
        max_prompt_length: Maximum length for prompts
        max_length: Maximum total sequence length  
        batch_size: Batch size for evaluation
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        **kwargs: Additional arguments passed to extract_logps
        
    Returns:
        DataFrame with evaluation results per sample
    """
    if isinstance(dataset, str):
        dataset_name, split = dataset.split("#")
        dataset: Dataset = load_dataset(dataset_name, split=split, keep_in_memory=False)

    model.eval()
    model.config.use_cache = False
    dataset_name = ds2name(dataset)

    # Validate length constraints
    assert max_length > max_prompt_length, (
        f"max_length {max_length} must be greater than max_prompt_length {max_prompt_length}"
    )

    tokenized_datasets = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        verbose=verbose>=1,
    )
    
    # Setup data collator and loader
    data_collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    evaluation_data = []

    for step, batch in enumerate(
        tqdm(eval_dataloader, desc=f"Eval {dataset_name}", disable=verbose < 2)
    ):
        if is_peft_model(model):
            # Loop through PEFT adapters if available
            if adapter_names is None:
                adapter_names: list = [None] + list(model.peft_config.keys())
                logger.debug(f"Detected adapters: {adapter_names}")
            
            for adapter_name in adapter_names:
                with set_adapter(model, adapter_name):
                    batch_results = extract_logps(model, batch, step, **kwargs)
                    adapter_name_clean = adapter_name if adapter_name is not None else "none"
                    batch_results["adapter"] = [adapter_name_clean] * len(batch_results["prob"])
                    evaluation_data.append(batch_results)
        else:
            # Standard model evaluation
            batch_results = extract_logps(model, batch, step, **kwargs)
            batch_results["adapter"] = ["none"] * len(batch_results["prob"])  # no adapter
            evaluation_data.append(batch_results)

    # Concatenate all batch results into a single DataFrame
    if not evaluation_data:
        logger.warning(f"No evaluation data collected for dataset {dataset_name}")
        return pd.DataFrame()
    
    # Combine all batch results
    combined_keys = evaluation_data[0].keys()
    combined_data = {key: itertools.chain(*[batch[key] for batch in evaluation_data]) 
                    for key in combined_keys}
    df = pd.DataFrame(combined_data)

    # Add derived columns
    df["correct"] = df["prob"] > 0.5
    df.fillna({'adapter': 'none'}, inplace=True)
    df["model"] = model.config._name_or_path + df['adapter']
    df["dataset"] = dataset_name

    return df


def eval_datasets(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    datasets: List[Dataset], 
    verbose: int = 1, 
    **kwargs
) -> pd.DataFrame:
    """Evaluate model on multiple datasets.
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model  
        datasets: List of datasets to evaluate
        verbose: Verbosity level for progress tracking
        **kwargs: Additional arguments passed to eval_dataset
        
    Returns:
        Combined DataFrame with results from all datasets
    """
    dfs = []
    for dataset in tqdm(datasets, disable=not verbose, unit="dataset"):
        df = eval_dataset(model, tokenizer, dataset, verbose=verbose, **kwargs)
        dfs.append(df)
        clear_mem()
    
    if not dfs:
        warnings.warn("No datasets processed")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    df["model"] = model.config._name_or_path  # Consistent model naming
    return df


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    datasets: List[Dataset],
    score_fn: Callable = score_ipo,
    verbose: int = 1,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a single model on multiple datasets with aggregated results.
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model
        datasets: List of datasets to evaluate on
        score_fn: Scoring function to use for evaluation
        verbose: Verbosity level for progress tracking
        **kwargs: Additional arguments passed to eval_datasets
        
    Returns:
        Tuple of (aggregated_results, raw_results) DataFrames
    """
    df_raw = eval_datasets(
        model, tokenizer, datasets, score_fn=score_fn, verbose=verbose, **kwargs
    )

    if df_raw.empty:
        logger.warning("No evaluation data collected")
        return pd.DataFrame(), pd.DataFrame()

    # Reorder columns for better readability
    cols = ["model", "dataset", "ds_i", "correct", "prob"]
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols + others]

    # Compute aggregated statistics
    numeric_cols = list(set(["correct", "prob"]).intersection(df_raw.columns))
    agg_cols = ["dataset"]
    if "adapter" in df_raw.columns:
        agg_cols.append("adapter")

    df_agg = df_raw.groupby(agg_cols, dropna=False)[numeric_cols].mean()
    df_agg["n"] = df_raw.groupby(agg_cols, dropna=False).size()
    clear_mem()

    # TODO temp scaling or logits
    return df_agg, df_raw


def evaluate_models(
    datasets: List[Dataset], 
    model_names: List[str], 
    model_kwargs: dict = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple models on multiple datasets.
    
    Args:
        datasets: List of datasets to evaluate on
        model_names: List of model names/paths to evaluate
        model_kwargs: Keyword arguments for model loading
        **kwargs: Additional arguments passed to evaluate_model
        
    Returns:
        Tuple of (aggregated_results, raw_results) DataFrames
    """
    if model_kwargs is None:
        model_kwargs = {}
        
    dfs_raw = []
    for model_name in model_names:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            df_agg, df_raw = evaluate_model(model, tokenizer, datasets=datasets, **kwargs)
            df_raw["model"] = model_name
            clear_mem()
            dfs_raw.append(df_raw)
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}")
            continue
    
    if not dfs_raw:
        logger.error("No models were successfully evaluated")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all raw results
    df_raw_combined = pd.concat(dfs_raw, ignore_index=True)
    
    # Keep only essential columns for final output
    essential_cols = ["model", "dataset", "ds_i", "correct", "prob"]
    available_cols = [col for col in essential_cols if col in df_raw_combined.columns]
    df_raw_final = df_raw_combined[available_cols]

    # Compute final aggregated results
    df_agg = df_raw_final.groupby(["model", "dataset"], dropna=False)[
        ["correct", "prob"]
    ].mean()

    return df_agg, df_raw_final


def evaluate(
    model_names: List[str], 
    datasets: Optional[List[Dataset]] = None, 
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main evaluation function for multiple models and datasets.
    
    Args:
        model_names: List of model names/paths to evaluate
        datasets: List of datasets to evaluate on (uses defaults if None)
        **kwargs: Additional arguments passed to evaluate_models
        
    Returns:
        Tuple of (aggregated_results, raw_results) DataFrames
    """
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(model_names=model_names, datasets=datasets, **kwargs)
