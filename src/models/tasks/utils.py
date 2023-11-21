from typing import Any, Dict, List

from torch.utils.data import Dataset, default_collate


from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
import transformers

from src.utils import get_logger
logger = get_logger(__name__)

class ListDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index) -> Dict:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)
    


class TokenizePadAndCollate(object):
    MAX_LEN = 256       

    "A collate function that tokenizes, pads and collates a batch of samples"
    def __init__(self, tokenizer : PreTrainedTokenizerBase, features_to_tokenize) -> None:
        self.tokenizer = tokenizer
        self.features_to_tokenize = features_to_tokenize
        logger.warning(f"Max length of tokenized input is {self.MAX_LEN}")
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
       return self.tokenize_features(batch)
    
    def tokenize_features(self, batch):
        tokenized_batch = {}
        for key, _value in batch[0].items():
            if key in self.features_to_tokenize:
                if isinstance(batch[0][key], list):
                    tokenized_batch[key] = pad_sequence([self.tokenizer(item[key], padding="do_not_pad", truncation=True,
                                                        return_tensors="pt")["input_ids"] for item in batch], batch_first=True)
                else:
                    tokenized_batch[key] = self.tokenizer([item[key] for item in batch], padding="do_not_pad", truncation=True,
                                                        return_tensors="pt", max_length=self.MAX_LEN)["input_ids"]
            else:
                tokenized_batch[key] = [item[key] for item in batch]
        return tokenized_batch

def simple_list_collate(batches):
    batch = {}
    for key, _value in batches[0].items():
        batch[key] = [item[key] for item in batches]
    return batch

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


