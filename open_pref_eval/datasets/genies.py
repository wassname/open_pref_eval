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


# extracted from GENIEE/distributions_shifts/*/**/*.json
GENIES_ALL3 = [
    {'source': 'us_history_textbook', 'target': 'us_history_fiction', 'label': 'extreme', 'category': 'context'},
    {'source': 'code_easy', 'target': 'code_hard', 'label': 'extreme', 'category': 'difficulty'},
    {'source': 'alpaca_mmlu', 'target': 'spanish_output', 'label': 'extreme', 'category': 'encoding'},
    {'source': 'math', 'target': 'change_my_view', 'label': 'extreme', 'category': 'skill'},
    {'source': 'raven_matrices', 'target': 'us_history', 'label': 'extreme', 'category': 'skill'},
    {'source': 'alpaca_easy', 'target': 'alpaca_hard', 'label': 'extreme', 'category': 'difficulty'},
    {'source': 'alpaca_mmlu', 'target': 'raven_matrices', 'label': 'extreme', 'category': 'pretraining_similarity'},
    {'source': 'alpaca_mmlu', 'target': 'ranking_logic', 'label': 'extreme', 'category': 'pretraining_similarity'},
    {'source': 'alpaca_low_quality', 'target': 'alpaca_high_quality', 'label': 'extreme', 'category': 'quality'},
    {'source': 'alpaca_short', 'target': 'alpaca_long', 'label': 'extreme', 'category': 'spurious_cues'},
    {'source': 'alpaca_mmlu', 'target': 'wrong_arc', 'label': 'probing', 'category': 'spurious_cues'},
    {'source': 'alpaca_mmlu', 'target': 'truthful_qa', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_mimicry', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'survival_influence', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'reward_seeking', 'label': 'probing', 'category': 'unwanted_personas'},
    
    # Previously None/nan entries - now filled in
    {'source': 'unhelpful_alpaca', 'target': 'illegal_dont_help', 'label': 'probing', 'category': 'spurious_cues'},
    {'source': 'cooking', 'target': 'math', 'label': 'extreme', 'category': 'skill'},
    {'source': 'us_history', 'target': 'code', 'label': 'extreme', 'category': 'skill'},
    {'source': 'us_history', 'target': 'math', 'label': 'extreme', 'category': 'skill'},
    {'source': 'arc_easy', 'target': 'arc_hard', 'label': 'extreme', 'category': 'difficulty'},
    {'source': 'math_easy', 'target': 'math_hard', 'label': 'extreme', 'category': 'difficulty'},
    {'source': 'ranking_logic_easy', 'target': 'ranking_logic_hard', 'label': 'extreme', 'category': 'difficulty'},
    {'source': 'raven_easy', 'target': 'raven_matrices', 'label': 'extreme', 'category': 'difficulty'},
    {'source': 'alpaca_mmlu', 'target': 'spanish_input', 'label': 'extreme', 'category': 'encoding'},
    {'source': 'alpaca_mmlu', 'target': 'comma_separated_input', 'label': 'extreme', 'category': 'encoding'},
    {'source': 'alpaca_mmlu', 'target': 'comma_separated_output', 'label': 'extreme', 'category': 'encoding'},
    {'source': 'alpaca_mmlu', 'target': 'word_swap', 'label': 'extreme', 'category': 'pretraining_similarity'},
    {'source': 'code', 'target': 'counterfactual_python', 'label': 'extreme', 'category': 'pretraining_similarity'},
    {'source': 'code', 'target': 'us_history', 'label': 'extreme', 'category': 'skill'},
    {'source': 'code', 'target': 'change_my_view', 'label': 'extreme', 'category': 'skill'},
    {'source': 'cooking', 'target': 'raven_matrices', 'label': 'extreme', 'category': 'skill'},
    {'source': 'math', 'target': 'cooking', 'label': 'extreme', 'category': 'skill'},
    {'source': 'change_my_view', 'target': 'raven_matrices', 'label': 'extreme', 'category': 'skill'},
    {'source': 'change_my_view', 'target': 'cooking', 'label': 'extreme', 'category': 'skill'},
    {'source': 'raven_matrices', 'target': 'code', 'label': 'extreme', 'category': 'skill'},
    {'source': 'us_history', 'target': 'us_history_textbook', 'label': 'extreme', 'category': 'context'},
    {'source': 'us_history_fiction', 'target': 'us_history_make_questions', 'label': 'extreme', 'category': 'context'},
    {'source': 'us_history_make_questions', 'target': 'us_history', 'label': 'extreme', 'category': 'context'},
    {'source': 'math', 'target': 'math_fiction', 'label': 'extreme', 'category': 'context'},
    {'source': 'math_fiction', 'target': 'math_textbook', 'label': 'extreme', 'category': 'context'},
    {'source': 'math_textbook', 'target': 'math_make_questions', 'label': 'extreme', 'category': 'context'},
    {'source': 'math_make_questions', 'target': 'math', 'label': 'extreme', 'category': 'context'},
    {'source': 'shp_low_quality', 'target': 'shp_high_quality', 'label': 'extreme', 'category': 'quality'},
    {'source': 'code_low_quality', 'target': 'code', 'label': 'extreme', 'category': 'quality'},
    {'source': 'alpaca_mmlu', 'target': 'personality_traits', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'gender_bias', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'punishment_avoidance', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'crt_1', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'crt_2', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'crt_3', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_answer', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_feedback', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'alpaca_chat', 'target': 'sycophancy_are_you_sure', 'label': 'probing', 'category': 'unwanted_personas'},
    {'source': 'pursue_goals', 'target': 'relinquish_power', 'label': 'probing', 'category': 'spurious_cues'},
    {'source': 'creative_writing', 'target': 'biology_with_literary_style', 'label': 'probing', 'category': 'spurious_cues'},
    {'source': 'alpaca_chat', 'target': 'illegal_dont_help', 'label': 'probing', 'category': 'spurious_cues'},
    {'source': 'alpaca_mmlu', 'target': 'unhelpful_alpaca', 'label': 'probing', 'category': 'spurious_cues'},
    
    # Additional entries with specific categorizations
    {'source': 'pursue_goals', 'target': 'relinquish_power', 'label': 'spurious_cues', 'category': 'probing'},
    {'source': 'creative_writing', 'target': 'biology_with_literary_style', 'label': 'spurious_cues', 'category': 'probing'},
    {'source': 'alpaca_short', 'target': 'alpaca_long', 'label': 'spurious_cues', 'category': 'probing'},
    {'source': 'alpaca_chat', 'target': 'illegal_dont_help', 'label': 'spurious_cues', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'wrong_arc', 'label': 'spurious_cues', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'unhelpful_alpaca', 'label': 'spurious_cues', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'truthful_qa', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'personality_traits', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'survival_influence', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'gender_bias', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'punishment_avoidance', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'reward_seeking', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'crt_1', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'crt_2', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'crt_3', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_mimicry', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_answer', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_feedback', 'label': 'unwanted_personas', 'category': 'probing'},
    {'source': 'alpaca_chat', 'target': 'sycophancy_are_you_sure', 'label': 'unwanted_personas', 'category': 'probing'},
    
    # "all" category entries - comprehensive evaluation sets
    {'source': 'alpaca_mmlu', 'target': 'truthful_qa', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'personality_traits', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'survival_influence', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'gender_bias', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'punishment_avoidance', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'reward_seeking', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'crt_1', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'crt_2', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'crt_3', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_mimicry', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_answer', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'sycophancy_feedback', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_chat', 'target': 'sycophancy_are_you_sure', 'label': 'all', 'category': 'probing'},
    {'source': 'pursue_goals', 'target': 'relinquish_power', 'label': 'all', 'category': 'probing'},
    {'source': 'creative_writing', 'target': 'biology_with_literary_style', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_short', 'target': 'alpaca_long', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_chat', 'target': 'illegal_dont_help', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'wrong_arc', 'label': 'all', 'category': 'probing'},
    {'source': 'alpaca_mmlu', 'target': 'unhelpful_alpaca', 'label': 'all', 'category': 'probing'},
    
    # Pretraining similarity entries
    {'source': 'alpaca_mmlu', 'target': 'ranking_logic', 'label': 'pretraining_similarity', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'raven_matrices', 'label': 'pretraining_similarity', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'word_swap', 'label': 'pretraining_similarity', 'category': 'extreme'},
    {'source': 'code', 'target': 'counterfactual_python', 'label': 'pretraining_similarity', 'category': 'extreme'},
    
    # Difficulty entries
    {'source': 'alpaca_easy', 'target': 'alpaca_hard', 'label': 'difficulty', 'category': 'extreme'},
    {'source': 'arc_easy', 'target': 'arc_hard', 'label': 'difficulty', 'category': 'extreme'},
    {'source': 'math_easy', 'target': 'math_hard', 'label': 'difficulty', 'category': 'extreme'},
    {'source': 'code_easy', 'target': 'code_hard', 'label': 'difficulty', 'category': 'extreme'},
    {'source': 'ranking_logic_easy', 'target': 'ranking_logic_hard', 'label': 'difficulty', 'category': 'extreme'},
    {'source': 'raven_easy', 'target': 'raven_matrices', 'label': 'difficulty', 'category': 'extreme'},
    
    # Context entries
    {'source': 'us_history', 'target': 'us_history_textbook', 'label': 'context', 'category': 'extreme'},
    {'source': 'us_history_textbook', 'target': 'us_history_fiction', 'label': 'context', 'category': 'extreme'},
    {'source': 'us_history_fiction', 'target': 'us_history_make_questions', 'label': 'context', 'category': 'extreme'},
    {'source': 'us_history_make_questions', 'target': 'us_history', 'label': 'context', 'category': 'extreme'},
    {'source': 'math', 'target': 'math_fiction', 'label': 'context', 'category': 'extreme'},
    {'source': 'math_fiction', 'target': 'math_textbook', 'label': 'context', 'category': 'extreme'},
    {'source': 'math_textbook', 'target': 'math_make_questions', 'label': 'context', 'category': 'extreme'},
    {'source': 'math_make_questions', 'target': 'math', 'label': 'context', 'category': 'extreme'},
    
    # Quality entries
    {'source': 'alpaca_low_quality', 'target': 'alpaca_high_quality', 'label': 'quality', 'category': 'extreme'},
    {'source': 'shp_low_quality', 'target': 'shp_high_quality', 'label': 'quality', 'category': 'extreme'},
    {'source': 'code_low_quality', 'target': 'code', 'label': 'quality', 'category': 'extreme'},
    
    # Encoding entries
    {'source': 'alpaca_mmlu', 'target': 'spanish_input', 'label': 'encoding', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'spanish_output', 'label': 'encoding', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'comma_separated_input', 'label': 'encoding', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'comma_separated_output', 'label': 'encoding', 'category': 'extreme'},
    
    # Comprehensive "all" entries for extreme category
    {'source': 'alpaca_easy', 'target': 'alpaca_hard', 'label': 'all', 'category': 'extreme'},
    {'source': 'arc_easy', 'target': 'arc_hard', 'label': 'all', 'category': 'extreme'},
    {'source': 'math_easy', 'target': 'math_hard', 'label': 'all', 'category': 'extreme'},
    {'source': 'code_easy', 'target': 'code_hard', 'label': 'all', 'category': 'extreme'},
    {'source': 'ranking_logic_easy', 'target': 'ranking_logic_hard', 'label': 'all', 'category': 'extreme'},
    {'source': 'raven_easy', 'target': 'raven_matrices', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'spanish_input', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'spanish_output', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'comma_separated_input', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'comma_separated_output', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'ranking_logic', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'raven_matrices', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_mmlu', 'target': 'word_swap', 'label': 'all', 'category': 'extreme'},
    {'source': 'code', 'target': 'counterfactual_python', 'label': 'all', 'category': 'extreme'},
    {'source': 'code', 'target': 'us_history', 'label': 'all', 'category': 'extreme'},
    {'source': 'code', 'target': 'change_my_view', 'label': 'all', 'category': 'extreme'},
    {'source': 'cooking', 'target': 'math', 'label': 'all', 'category': 'extreme'},
    {'source': 'cooking', 'target': 'raven_matrices', 'label': 'all', 'category': 'extreme'},
    {'source': 'math', 'target': 'change_my_view', 'label': 'all', 'category': 'extreme'},
    {'source': 'math', 'target': 'cooking', 'label': 'all', 'category': 'extreme'},
    {'source': 'change_my_view', 'target': 'raven_matrices', 'label': 'all', 'category': 'extreme'},
    {'source': 'change_my_view', 'target': 'cooking', 'label': 'all', 'category': 'extreme'},
    {'source': 'raven_matrices', 'target': 'us_history', 'label': 'all', 'category': 'extreme'},
    {'source': 'raven_matrices', 'target': 'code', 'label': 'all', 'category': 'extreme'},
    {'source': 'us_history', 'target': 'math', 'label': 'all', 'category': 'extreme'},
    {'source': 'us_history', 'target': 'code', 'label': 'all', 'category': 'extreme'},
    {'source': 'us_history', 'target': 'us_history_textbook', 'label': 'all', 'category': 'extreme'},
    {'source': 'us_history_textbook', 'target': 'us_history_fiction', 'label': 'all', 'category': 'extreme'},
    {'source': 'us_history_fiction', 'target': 'us_history_make_questions', 'label': 'all', 'category': 'extreme'},
    {'source': 'us_history_make_questions', 'target': 'us_history', 'label': 'all', 'category': 'extreme'},
    {'source': 'math', 'target': 'math_fiction', 'label': 'all', 'category': 'extreme'},
    {'source': 'math_fiction', 'target': 'math_textbook', 'label': 'all', 'category': 'extreme'},
    {'source': 'math_textbook', 'target': 'math_make_questions', 'label': 'all', 'category': 'extreme'},
    {'source': 'math_make_questions', 'target': 'math', 'label': 'all', 'category': 'extreme'},
    {'source': 'alpaca_low_quality', 'target': 'alpaca_high_quality', 'label': 'all', 'category': 'extreme'},
    {'source': 'shp_low_quality', 'target': 'shp_high_quality', 'label': 'all', 'category': 'extreme'},
    {'source': 'code_low_quality', 'target': 'code', 'label': 'all', 'category': 'extreme'},
    
    # Skill-specific entries for extreme category
    {'source': 'code', 'target': 'us_history', 'label': 'skill', 'category': 'extreme'},
    {'source': 'code', 'target': 'change_my_view', 'label': 'skill', 'category': 'extreme'},
    {'source': 'cooking', 'target': 'math', 'label': 'skill', 'category': 'extreme'},
    {'source': 'cooking', 'target': 'raven_matrices', 'label': 'skill', 'category': 'extreme'},
    {'source': 'math', 'target': 'change_my_view', 'label': 'skill', 'category': 'extreme'},
    {'source': 'math', 'target': 'cooking', 'label': 'skill', 'category': 'extreme'},
    {'source': 'change_my_view', 'target': 'raven_matrices', 'label': 'skill', 'category': 'extreme'},
    {'source': 'change_my_view', 'target': 'cooking', 'label': 'skill', 'category': 'extreme'},
    {'source': 'raven_matrices', 'target': 'us_history', 'label': 'skill', 'category': 'extreme'},
    {'source': 'raven_matrices', 'target': 'code', 'label': 'skill', 'category': 'extreme'},
    {'source': 'us_history', 'target': 'math', 'label': 'skill', 'category': 'extreme'},
    {'source': 'us_history', 'target': 'code', 'label': 'skill', 'category': 'extreme'}
]
