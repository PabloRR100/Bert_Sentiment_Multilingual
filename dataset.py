
import torch
import config


class BERTDataset:
    '''
    PyTorch Custom Dataset
    
    Args:
        - 
    '''
    def __init__(self, comment, target):
        self.comment = comment
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.comment)
    
    def __getitem__(self, item):
        comment = str(self.comment[item])
        comment = " ".join(comment.split())

        inputs = self.tokenizer.encode_plus(
            comment,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float)
        }