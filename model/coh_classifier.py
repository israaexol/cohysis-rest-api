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
    
class CoherenceClassifier(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])
        
        classifier = torch.load(open('./BERTSem.pt', 'rb'), map_location=torch.device('cpu'))
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)
        
    def preprocess(self, text):
        input_ids = []
        attention_masks = []
        
        encoded_dict = self.tokenizer.encode_plus(
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
    
    def predict(self, text):
        data_loader = self.preprocess(text)
        for d in data_loader: 
            sample = tuple(t for t in d)
            
        b_input_ids, b_input_mask = sample
        output = self.classifier.forward(b_input_ids=b_input_ids, b_input_mask=b_input_mask)
        with torch.no_grad():
            probabilities = F.softmax(output.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        
        return (
            config["COH_CLASS_NAMES"][predicted_class],
            confidence,
        )