# load the pretrained model (pt file)
# define a function to preprocess the data
# define a predict function that assembles these two

import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

with open("config.json") as json_file:
    config = json.load(json_file)
    
def preprocess(package: dict, text: str) -> dict:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding
    :param package: dict from fastapi state including model and preocessing objects
    :param text: text to be proprocessed
    :return: dictionary of processed text
    """

    # tokenize the input text
    tokenizer = package['tokenizer']
    input_ids = []
    attention_masks = []
    
    encoded_dict = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    sample = TensorDataset(input_ids, attention_masks)
    sample_loader = DataLoader(
            sample,
            sampler = RandomSampler(sample),
            batch_size = 1
        )
    
    return sample_loader


def predict(package: dict, data_loader: dict) -> str:
    for d in data_loader: 
        sample = tuple(t for t in d)
        
    model = package['model']
    b_input_ids, b_input_mask = sample
    output = model.forward(input_ids=b_input_ids, attention_mask=b_input_mask)
    with torch.no_grad():
        probabilities = F.softmax(output.logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    predicted_class = predicted_class.cpu().item()
    
    return config["COH_CLASS_NAMES"][predicted_class]