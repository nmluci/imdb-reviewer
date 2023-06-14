import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import numpy as np

sentichan = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', lower=True)

sentichan.load_state_dict(torch.load("sentichan_v2_state.pt", map_location=torch.device('cpu')))
sentichan.eval()

for param in sentichan.state_dict():
    print(param, "\t", sentichan.state_dict()[param].size())

# with torch.no_grad():
#    input_text = "this review is perfectly majestic"
#    encoded = tokenizer(input_text, truncation=True)

#    text = torch.tensor(encoded["input_ids"]).unsqueeze(0)
#    attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0)

#    output = sentichan(text, attention_mask)

