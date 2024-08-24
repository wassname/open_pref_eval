from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, get_peft_model, PeftConfig, PeftModelForCausalLM

def load_peft_model(model_name):
    peft_config = PeftConfig.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModelForCausalLM.from_pretrained(
        base_model,
        model_name, config=peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
