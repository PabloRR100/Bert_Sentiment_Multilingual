
import torch
import torch.nn as nn
import config
import transformers


"""
EXTENDES TRANSFORMERS BERT MODEL
"""


class BERTBaseUncased(nn.Module):

    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(2*768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        o1, _ = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        mean_pool = torch.mean(o1,1)
        max_pool,_ = torch.max(o1,1)
        cat = torch.cat((mean_pool, max_pool),dim=1)
        bo = self.bert_drop(cat)
        output = self.out(bo)
        return output