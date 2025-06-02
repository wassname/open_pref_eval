# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import AutoPeftModelForCausalLM, get_peft_model, PeftConfig, PeftModelForCausalLM

# def load_peft_model(model_name, load_4bit=False, load_8bit=False):

#     if load_4bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#         )
#     elif load_8bit:
#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#         )
#     else:
#         quantization_config = None
    
#     # TODO detect if it's peft from the config
#     # TODO pass in kwargs
#     # TODO allow 8bit
#     try:
#         peft_config = PeftConfig.from_pretrained(model_name)
#         base_model_name = peft_config.base_model_name_or_path
#         tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#     except Exception as e:
#         raise # FIXME specific error

#     model = PeftModelForCausalLM.from_pretrained(
#         base_model,
#         model_name, 
#         low_cpu_mem_usage=True,
#         # torch_dtype=torch.bfloat16,
#         # attn_implementation="flash_attention_2",
#         config=peft_config)


#     if hasattr(model, 'peft_config'):
#         print("PeftConfig loaded:", model.peft_config)
#         # Wrap as Peft model if not already wrapped
#         if not isinstance(model, PeftModelForCausalLM):
#             model = get_peft_model(model, peft_config)

#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#     except Exception as e:
#         print(f"Failed to load tokenizer for {model_name}: {e}")
#         raise # TODO make specific error 
#         # Fallback to the base model's tokenizer if the peft model's tokenizer fails
#         peft_config = PeftConfig.from_pretrained(model_name)
#         if 'base_model_name_or_path' in peft_config:
#             tokenizer = AutoTokenizer.from_pretrained(peft_config['base_model_name_or_path'])
#     return model, tokenizer
