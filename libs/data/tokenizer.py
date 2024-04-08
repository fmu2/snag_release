from copy import deepcopy

import torchtext
from torchtext.data import get_tokenizer


tokenizers = dict()
def register_tokenizer(name):
    def decorator(module):
        tokenizers[name] = module
        return module
    return decorator


@register_tokenizer('glove')
class GloVeTokenizer:

    def __init__(self, name='6B'):

        self.vocab = torchtext.vocab.GloVe(name=name)
        self.tokenizer = get_tokenizer("basic_english")

    def __call__(self, text, max_len=None):
        """
        Args:
            text (str): text query.
            max_len (int): maximum sequence length.

        Returns:
            feats (float tensor, (c, t)): feature sequence.
        """
        # tokenize by word
        ## NOTE: unknown words are assigned zero vector
        words = self.tokenizer(text)
        feats = self.vocab.get_vecs_by_tokens(words, lower_case_backup=True)
        if max_len is not None:
            feats = feats[:max_len]
        feats = feats.transpose(0, 1)   # (c, t)

        return feats


def make_tokenizer(name):
    return tokenizers[name]()
