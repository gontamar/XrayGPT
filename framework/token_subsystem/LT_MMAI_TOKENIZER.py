from transformers import AutoTokenizer
#from xraygpt.models.blip2 import Blip2Base

# For BLIP2 (BERT based)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})

input_text = "This is a sample radiology report for testing."
tokens = bert_tokenizer.tokenize(input_text)
token_ids = bert_tokenizer.encode(input_text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

