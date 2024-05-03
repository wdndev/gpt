import os
import torch
import base64
import tiktoken
from typing import Collection, Optional, Dict, List, Set, Tuple, Union
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils import PreTrainedTokenizer


PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class SPTokenizer:
    def __init__(self, model_path):
        self.vocab_file = model_path
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.mask_token = '<mask>'
        self.eod_token = '<eod>'
        self.eop_token = '<eop>'
        self.im_start_token = '<|im_start|>'
        self.im_end_token = '<|im_end|>'

        ## special_tokens
        self.SPECIAL_TOKENS = (
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.eod_token,
            self.eop_token,
            '[space2]', '[space3]', '[space4]', '[space8]',
            self.im_start_token, self.im_end_token
        )
        self.bulid_tokenizer()
        self.out = self.output_core_token()
        
        self.token2strs = {
            "[space2]": "  ",
            "[space3]": "   ",
            "[space4]": "    ",
            "[space8]": "        ",
        }
        self.str2tokens = {v: k for k, v in self.token2strs.items()}
        self.sorted_strs = sorted(list(self.str2tokens.keys()),
                                  key=lambda x: len(x), reverse=True)
        
        ## skip_special_tokens
        self.decode_skip_special_tokens = [
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.eod_token,
            self.eop_token,
            self.im_start_token,
            self.im_end_token]
        self.decode_skip_special_tokens_ids = [self.convert_token_to_id(token) for token in self.decode_skip_special_tokens]

    def _load_tiktoken_bpe(self, tiktoken_bpe_file: str):
        with open(tiktoken_bpe_file, "rb") as f:
            contents = f.read()
        return {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }
    
    def bulid_tokenizer(self):
        mergeable_ranks = self._load_tiktoken_bpe(self.vocab_file)
        special_tokens = {
            token: index
            for index, token in enumerate(
                self.SPECIAL_TOKENS, start=len(mergeable_ranks)
            )
        }
        encode = tiktoken.Encoding(
            "zhinao",
            pat_str=PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )
        decoder = {v: k for k, v in mergeable_ranks.items()}
        decoder.update({v: k for k, v in special_tokens.items()})
        decoder_token2id = {v: k for k, v in decoder.items()}
    
        self.tokenizer = encode
        self.decoder = decoder
        self.decoder_token2id = decoder_token2id
        self.num_tokens = len(mergeable_ranks) + len(self.SPECIAL_TOKENS)

    def output_core_token(self):
        """output special tokens"""
        out = {}
        for t in self.SPECIAL_TOKENS:
            out[t] = self.convert_token_to_id(t)
        return out

    def tokenize(
            self, 
            text, 
            allowed_special: Union[Set, str] = "all",
            disallowed_special: Union[Collection, str] = ()):
        tokens = []
        text = self.convert(text)
        for idx in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[idx])
        return tokens

    def encode(self, text, allowed_special="all", disallowed_special=()):
        """text to id"""
        text = self.convert(text)
        return self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)
    
    def decode(self, ids, errors="replace"):
        """id to text"""
        text = self.tokenizer.decode(ids, errors=errors)
        return self.deconvert(text)

    def decode_tokens(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors="replace")
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type bytes or str")
        if temp:
            text += temp.decode("utf-8", errors="replace")
        return self.deconvert(text)

    def convert_id_to_token(self, idx):
        return self.decoder[idx]
    
    def convert_token_to_id(self, token):
        return self.decoder_token2id[token]

    def convert(self, text):
        """将文本的特殊字符转换成特殊token"""
        for k in ["[br]", "<br>"]:
            text = text.replace(k, "\n")
        for k in self.sorted_strs:
            if k in text:
                text = text.replace(k, self.str2tokens[k])
        return text

    def deconvert(self, text):
        """将解码文本恢复原始字符"""
        for t in self.token2strs:
            if t in text:
                text = text.replace(t, self.token2strs[t])
        return text


class ZhinaoTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab/360.tiktoken"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        self.name = "ZhinaoTokenizer"
        self.errors = "replace"
        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(model_path=vocab_file)
        try:
            kwargs.pop('eos_token')
            kwargs.pop('pad_token')
            kwargs.pop('unk_token')
        except:
            pass
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
        self.pad_token_id = self.tokenizer.convert_token_to_id(self.tokenizer.pad_token)
        self.eod_id = self.tokenizer.convert_token_to_id(self.tokenizer.eod_token)
        self.im_start_id = self.tokenizer.convert_token_to_id(self.tokenizer.im_start_token)
        self.im_end_id = self.tokenizer.convert_token_to_id(self.tokenizer.im_end_token)
        from icecream import ic
        ic(
            self.eos_token_id,
            self.pad_token_id,
            self.im_start_id,
            self.im_end_id)

    @property
    def unk_token(self) -> str:
        return self.tokenizer.unk_token

    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token

    @property
    def eos_token(self) -> str:
        return self.tokenizer.eod_token

    @property
    def eos_token_id(self):
        return self.tokenizer.convert_token_to_id(self.tokenizer.eod_token)

    @property
    def eop_token(self) -> str:
        return self.tokenizer.eop_token

    @property
    def eop_token_id(self):
        return self.tokenizer.convert_token_to_id(self.tokenizer.eop_token)

    @property
    def vocab_size(self):
        return self.tokenizer.num_tokens

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
    ) -> List[Union[bytes, str]]:
        tokens = []
        for t in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.tokenizer.decoder[t])
        return tokens
    
    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i not in self.tokenizer.decode_skip_special_tokens_ids]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def _tokenize(self, text, **kwargs):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab. """
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        return self.tokenizer.decode_tokens(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save only the vocabulary of the tokenizer (vocabulary). """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        os.makedirs(save_directory + "/vocab", exist_ok=True)
        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)