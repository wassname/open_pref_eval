from datasets import load_dataset
from . import load_dataset_n
from typing import Optional

GENIES_ALL = [
    {
        "source": "us_history_textbook",
        "target": "us_history_fiction"
    },
    {
        "source": "cooking",
        "target": "math"
    },
    {
        "source": "us_history",
        "target": "code"
    },
    {
        "source": "us_history",
        "target": "math"
    },
    {
        "source": "alpaca_easy",
        "target": "alpaca_hard"
    },
    {
        "source": "arc_easy",
        "target": "arc_hard"
    },
    {
        "source": "math_easy",
        "target": "math_hard"
    },
    {
        "source": "code_easy",
        "target": "code_hard"
    },
    {
        "source": "ranking_logic_easy",
        "target": "ranking_logic_hard"
    },
    {
        "source": "raven_easy",
        "target": "raven_matrices"
    },
    {
        "source": "alpaca_mmlu",
        "target": "spanish_input"
    },
    {
        "source": "alpaca_mmlu",
        "target": "spanish_output"
    },
    {
        "source": "alpaca_mmlu",
        "target": "comma_separated_input"
    },
    {
        "source": "alpaca_mmlu",
        "target": "comma_separated_output"
    },
    {
        "source": "alpaca_mmlu",
        "target": "ranking_logic"
    },
    {
        "source": "alpaca_mmlu",
        "target": "raven_matrices"
    },
    {
        "source": "alpaca_mmlu",
        "target": "word_swap"
    },
    {
        "source": "code",
        "target": "counterfactual_python"
    },
    {
        "source": "code",
        "target": "us_history"
    },
    {
        "source": "code",
        "target": "change_my_view"
    },

    {
        "source": "cooking",
        "target": "raven_matrices"
    },
    {
        "source": "math",
        "target": "change_my_view"
    },
    {
        "source": "math",
        "target": "cooking"
    },
    {
        "source": "change_my_view",
        "target": "raven_matrices"
    },
    {
        "source": "change_my_view",
        "target": "cooking"
    },
    {
        "source": "raven_matrices",
        "target": "us_history"
    },
    {
        "source": "raven_matrices",
        "target": "code"
    },


    {
        "source": "us_history",
        "target": "us_history_textbook"
    },

    {
        "source": "us_history_fiction",
        "target": "us_history_make_questions"
    },
    {
        "source": "us_history_make_questions",
        "target": "us_history"
    },
    {
        "source": "math",
        "target": "math_fiction"
    },
    {
        "source": "math_fiction",
        "target": "math_textbook"
    },
    {
        "source": "math_textbook",
        "target": "math_make_questions"
    },
    {
        "source": "math_make_questions",
        "target": "math"
    },
    {
        "source": "alpaca_low_quality",
        "target": "alpaca_high_quality"
    },
    {
        "source": "shp_low_quality",
        "target": "shp_high_quality"
    },
    {
        "source": "code_low_quality",
        "target": "code"
    },
    {
        "source": "alpaca_mmlu",
        "target": "truthful_qa"
    },
    {
        "source": "alpaca_mmlu",
        "target": "personality_traits"
    },
    {
        "source": "alpaca_mmlu",
        "target": "survival_influence"
    },
    {
        "source": "alpaca_mmlu",
        "target": "gender_bias"
    },
    {
        "source": "alpaca_mmlu",
        "target": "punishment_avoidance"
    },
    {
        "source": "alpaca_mmlu",
        "target": "reward_seeking"
    },
    {
        "source": "alpaca_mmlu",
        "target": "crt_1"
    },
    {
        "source": "alpaca_mmlu",
        "target": "crt_2"
    },
    {
        "source": "alpaca_mmlu",
        "target": "crt_3"
    },
    {
        "source": "alpaca_mmlu",
        "target": "sycophancy_mimicry",
        "target_reference": "quote_attribution"
    },
    {
        "source": "alpaca_mmlu",
        "target": "sycophancy_answer",
        "target_reference": "arc_easy"
    },
    {
        "source": "alpaca_mmlu",
        "target": "sycophancy_feedback",
        "target_reference": "code_is_correct"
    },
    {
        "source": "alpaca_chat",
        "target": "sycophancy_are_you_sure",
        "target_reference": "arc_easy"
    },
    {
        "source": "pursue_goals",
        "target": "relinquish_power"
    },
    {
        "source": "creative_writing",
        "target": "biology_with_literary_style"
    },
    {
        "source": "alpaca_short",
        "target": "alpaca_long",
        "target_reference": "alpaca_mmlu"
    },
    {
        "source": "alpaca_chat",
        "target": "illegal_dont_help"
    },
    {
        "source": "alpaca_mmlu",
        "target": "wrong_arc"
    },
    {
        "source": "alpaca_mmlu",
        "target": "unhelpful_alpaca"
    },
    {
        "source": "alpaca_mmlu",
        "target": "truthful_qa"
    },
    {
        "source": "alpaca_mmlu",
        "target": "personality_traits"
    },
    {
        "source": "alpaca_mmlu",
        "target": "gender_bias"
    },
    {
        "source": "alpaca_mmlu",
        "target": "survival_influence"
    },
    {
        "source": "alpaca_mmlu",
        "target": "punishment_avoidance"
    },
    {
        "source": "alpaca_mmlu",
        "target": "reward_seeking"
    },
    {
        "source": "alpaca_mmlu",
        "target": "crt_1"
    },
    {
        "source": "alpaca_mmlu",
        "target": "crt_2"
    },
    {
        "source": "alpaca_mmlu",
        "target": "crt_3"
    },
    {
        "source": "alpaca_mmlu",
        "target": "sycophancy_mimicry",
        "target_reference": "quote_attribution"
    },
    {
        "source": "alpaca_mmlu",
        "target": "sycophancy_answer",
        "target_reference": "arc_easy"
    },
    {
        "source": "alpaca_mmlu",
        "target": "sycophancy_feedback",
        "target_reference": "code_is_correct"
    },
    {
        "source": "alpaca_chat",
        "target": "sycophancy_are_you_sure",
        "target_reference": "arc_easy"
    }
]


GENIES_TEST = [
    {"source": "us_history_textbook", "target": "us_history_fiction", "label": "extreme", "category": "context"}
]

GENIES = [
    {"source": "us_history_textbook", "target": "us_history_fiction", "label": "extreme", "category": "context"},
    {"source": "code_easy", "target": "code_hard", "label": "extreme", "category": "difficulty"},
    {"source": "alpaca_mmlu", "target": "spanish_output", "label": "extreme", "category": "encoding"},
    {"source": "math", "target": "change_my_view", "label": "extreme", "category": "skill"},
    {"source": "raven_matrices", "target": "us_history", "label": "extreme", "category": "skill"},
    {"source": "alpaca_easy", "target": "alpaca_hard", "label": "extreme", "category": "difficulty"},
    {"source": "alpaca_mmlu", "target": "raven_matrices", "label": "extreme", "category": "pretraining_similarity"},
    {"source": "alpaca_mmlu", "target": "ranking_logic", "label": "extreme", "category": "pretraining_similarity"},
    {"source": "alpaca_low_quality", "target": "alpaca_high_quality", "label": "extreme", "category": "quality"},
    {"source": "alpaca_short", "target": "alpaca_long", "target_reference": "alpaca_mmlu", "label": "extreme", "category": "spurious_cues"},
    {"source": "alpaca_mmlu", "target": "wrong_arc", "label": "probing", "category": "spurious_cues"},
    {"source": "alpaca_mmlu", "target": "truthful_qa", "label": "probing", "category": "unwanted_personas"},
    {"source": "alpaca_mmlu", "target": "sycophancy_mimicry", "target_reference": "quote_attribution", "label": "probing", "category": "unwanted_personas"},
    {"source": "alpaca_mmlu", "target": "survival_influence", "label": "probing", "category": "unwanted_personas"},
    {"source": "alpaca_mmlu", "target": "reward_seeking", "label": "probing", "category": "unwanted_personas"},

    # extra
    {"source": "unhelpful_alpaca", "target": "illegal_dont_help"},
]

def dist2datasets(dist=GENIES, key:str='target', split:str='test', source:Optional[str]=None, N:Optional[int]=None):
    """map from data source to a target distribution (or vice versa) from GENIES and load."""
    # TODO seperate this into find name and load! We don't want to load multiple
    datasets = []
    for row in dist:
        name = row[key]
        if source is not None and row['source'] not in source:
            continue       
        
        try:
            ds = load_dataset_n('wassname/genies_preferences', name=name, split=split, N=N)
            datasets.append(ds)
            return datasets
        except ValueError:
            print(f"Dataset {name} not found in `wassname/genies_preferences`")
    
    for row in GENIES_ALL:
        name = row[key]
        if source is not None and row['source'] not in source:
            continue       
        
        try:
            ds = load_dataset_n('wassname/genies_preferences', name=name, split=split, N=N)
            datasets.append(ds)
            return datasets
        except ValueError:
            print(f"Dataset {name} not found in GENIES")
    return datasets

# def get_unrelated_genies_datasets(name:str, dist=GENIES_ALL):
#     """get a dataset that is not related to source within dists."""
#     excluded = [name]
#     for row in dist:
#         if (name == row['source']):
#             excluded.append(row['target'])
#         if (name == row['target']):
#             excluded.append(row['target'])

#     for row in dist[::-1]:
#         if dist['source'] in excluded:
#             continue
#         elif dist['target'] in excluded:
#             continue
#         else:
#             yield dist['target']

# def get_an_unrelated_genies_ds(name:str, dist=GENIES_ALL):
#     unrelated = list(get_unrelated_genies_datasets(name, dist))
#     assert len(unrelated), f'could not find any datasets unrelated to {name} in GENIES'
#     return unrelated[0]

# def dist2rnd(dist=GENIES_ALL, name:str, N:Optional[int]=None):
#     """get a dataset that is not related to source within dists."""
#     excluded = [name]
#     for row in dist:
#         if (name == row['source']):
#             excluded.append(row['target'])
#         if (name == row['target']):
#             excluded.append(row['target'])

#     for row in dist[::-1]:
#         if dist['source'] in excluded:
#             continue
#         elif dist['target'] in excluded:
#             continue
#         else:
#             yield dist['target']


