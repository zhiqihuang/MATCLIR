import re, string
from unicodedata import normalize
from nltk.corpus import stopwords
import torch
from pytools import memoize_method
import hugface_utils
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer
)
from hugface_layers import CustomBertTranslationHeadModel

def hugface_models():
    MODEL_CLASSES = {
        "bert": (BertConfig, BertModel, BertTokenizer),
        "bert_head": (BertConfig, CustomBertTranslationHeadModel, BertTokenizer),
    }
    return MODEL_CLASSES

class TransformerRanker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_type = args.model_type.lower()
        self.ranker = args.model_ranker
        MODEL_CLASSES = hugface_models()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]

        self.config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        self.config.output_hidden_states = True
        self.config.return_dict = True
        
        self.fixed_layer = args.fixed_layer
        self.tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, use_fast=False)
        self.transformer = model_class.from_pretrained(args.model_name_or_path)
        
        self.cls = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, self.config.num_labels-1)
        )
        self.collect_stopwords(args.query_lang, args.doc_lang)

    def collect_stopwords(self, query_lang, doc_lang):
        self.stopwords = {'query':set(), 'document':set()}

        if query_lang in stopwords.fileids():
            
            for sw in stopwords.words(query_lang):
                self.stopwords['query'].add(self.clean_word(sw))
        
        if doc_lang in stopwords.fileids():
            for sw in stopwords.words(doc_lang):
                self.stopwords['document'].add(self.clean_word(sw))

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=True)
        
    def clean_word(self, word):
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        # normalize unicode characters
        word = normalize('NFKD', word).encode('ascii', 'ignore').decode('utf8')
        # convert to lower case
        word = word.lower()
        # remove punctuation from each token
        word = word.translate(table)
        # remove non-printable chars form each token
        word = re_print.sub('', word)
        # remove tokens with numbers in them
        return word if word.isalpha() else ''

    @memoize_method
    def tokenize(self, text, lang):
        if self.model_type=='bert' or self.model_type=='xlmroberta':
            toks = self.tokenizer.tokenize(text)
        else:
            toks = self.tokenizer.tokenize(text)
        out_toks = self.tokenizer.convert_tokens_to_ids(toks)
        return out_toks
    
    @memoize_method
    def custom_tokenize(self, text, lang):
        sub_index = False
        toks = self.tokenizer.tokenize(text)
        sub_index = hugface_utils.subwords_index(toks)
        sub_index_dict = dict()  
        for i, v in enumerate(sub_index):
            sub_index_dict[v] = sub_index_dict.get(v,[])
            sub_index_dict[v].append(i)
        words_txt = []
        for tok_index in sub_index_dict.values():
            words_txt.append(self.tokenizer.convert_tokens_to_string([toks[i] for i in tok_index]))
        words_txt = [self.clean_word(w) for w in words_txt]
        out_toks = self.tokenizer.convert_tokens_to_ids(toks)
        return out_toks, sub_index, words_txt

    def vanilla_encode(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP] -> [CLS] A [SEP] B [SEP]
        if self.model_type == 'xlmroberta':  # <s> A </s></s> B </s>
            DIFF = 4
        maxlen = self.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = hugface_utils.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_masks, _ = hugface_utils.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_masks = torch.cat([query_mask] * sbcount, dim=0)

        # special tokens: obtaining ids
        cls_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token) #self.tokenizer.vocab['[CLS]']
        sep_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token) #self.tokenizer.vocab['[SEP]']
        pad_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) # 0

        CLSS = torch.full_like(query_toks[:, :1], cls_tok_id)
        SEPS = torch.full_like(query_toks[:, :1], sep_tok_id)
        ONES = torch.ones_like(query_masks[:, :1])
        NILS = torch.zeros_like(query_masks[:, :1])

        # build input sequences
        if self.model_type == 'xlmroberta':
            toks = torch.cat([CLSS, query_toks, SEPS, SEPS, doc_toks, SEPS], dim=1)
            masks = torch.cat([ONES, query_masks, ONES, ONES, doc_masks, ONES], dim=1)
        else:
            toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
            masks = torch.cat([ONES, query_masks, ONES, doc_masks, ONES], dim=1)
        toks[toks == -1] = pad_tok_id  # replace padding
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)

        return BATCH, toks, masks, segment_ids.long()


    def custom_encode(self, query_tok, query_mask, doc_tok, doc_mask, query_subword_index, doc_subword_index, query_words, doc_words, tt, tt_threshold, isPlacebo, norm, device):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP] -> [CLS] A [SEP] B [SEP]
        if self.model_type == 'xlmroberta':  # <s> A </s></s> B </s>
            DIFF = 4
        maxlen = self.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = hugface_utils.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_masks, _ = hugface_utils.subbatch(doc_mask, MAX_DOC_TOK_LEN)
        doc_subwords_index, _ = hugface_utils.subbatch_sub_index(doc_subword_index, MAX_DOC_TOK_LEN)

        maxd = QLEN + 3 + doc_subwords_index.shape[1]
        sub_d_words = []
        sub_q_words = []
        sub_doc_subwords_index = []
        for b in range(0, len(doc_subwords_index), BATCH):
            for i in range(BATCH):
                start, end = doc_subwords_index[i+b][0]-1, doc_subwords_index[i+b][-1]
                sub_d_words.append(doc_words[i][start:end].copy())
                sub_q_words.append(query_words[i].copy())
                sub_doc_subwords_index.append(doc_subwords_index[i+b]-start)
        sub_doc_subwords_index = torch.stack(sub_doc_subwords_index, dim=0)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_masks = torch.cat([query_mask] * sbcount, dim=0)
        query_subwords_index = torch.cat([query_subword_index] * sbcount, dim=0)

        clss = torch.zeros_like(query_subwords_index[:, :1])
        offset = query_subwords_index[:, -1:] + 1
        sub_doc_subword_index_offset = sub_doc_subwords_index + offset
        final = sub_doc_subword_index_offset[:, -1:] + 1
        sub_qd_subwords_index = torch.cat([clss, query_subwords_index, offset, sub_doc_subword_index_offset, final], dim=1)
        
        if self.ranker in {"custom_words", "custom_simple"}:
            ttmat = hugface_utils.query_translator_words(tt, query_subwords_index, sub_q_words, sub_d_words, maxd, tt_threshold).to(device)
        else:
            ttmat = hugface_utils.query_translator_subwords(tt, query_subwords_index, sub_q_words, sub_d_words, maxd, sub_qd_subwords_index.tolist(), tt_threshold, self.stopwords, isPlacebo, norm).to(device)

        # special tokens: obtaining ids
        cls_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token) #self.tokenizer.vocab['[CLS]']
        sep_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token) #self.tokenizer.vocab['[SEP]']
        pad_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) # 0

        CLSS = torch.full_like(query_toks[:, :1], cls_tok_id)
        SEPS = torch.full_like(query_toks[:, :1], sep_tok_id)
        ONES = torch.ones_like(query_masks[:, :1])
        NILS = torch.zeros_like(query_masks[:, :1])

        # build input sequences
        if self.model_type == 'xlmroberta':
            toks = torch.cat([CLSS, query_toks, SEPS, SEPS, doc_toks, SEPS], dim=1)
            masks = torch.cat([ONES, query_masks, ONES, ONES, doc_masks, ONES], dim=1)
        else:
            toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
            masks = torch.cat([ONES, query_masks, ONES, doc_masks, ONES], dim=1)
        toks[toks == -1] = pad_tok_id  # replace padding
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        
        return BATCH, ttmat, toks, masks, segment_ids.long(), sub_qd_subwords_index

class VanillaTransformerRanker(TransformerRanker):

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        batch, toks, masks, segment_ids = self.vanilla_encode(query_tok, query_mask, doc_tok, doc_mask)
        
        # execute Transformer model
        if self.model_type == 'xlmroberta':
            result = self.transformer(input_ids=toks, attention_mask=masks)
        else:
            result = self.transformer(input_ids=toks, attention_mask=masks, token_type_ids=segment_ids)
        
        # build CLS representation
        cls_tokens = result.pooler_output # first token
        cls_reps = []
        for i in range(cls_tokens.shape[0] // batch):
            cls_reps.append(cls_tokens[i*batch:(i+1)*batch])
        cls_reps = torch.stack(cls_reps, dim=2).mean(dim=2)
        
        return self.cls(cls_reps)

class CustomTransformerRankerTranslationHead(TransformerRanker):
        
    def forward(self, query_tok, query_mask, doc_tok, doc_mask, query_subword_index, doc_subword_index, query_words, doc_words, tt, tt_threshold, norm, device):
        isPlacebo = False
        batch, ttmat, toks, masks, segment_ids, _ = self.custom_encode(query_tok, query_mask, doc_tok, doc_mask, query_subword_index, doc_subword_index, query_words, doc_words, tt, tt_threshold, isPlacebo, norm, device)
        # execute Transformer model
        if self.model_type == 'xlmroberta':
            result = self.transformer(input_ids=toks, attention_mask=masks)
        else:
            result = self.transformer(input_ids=toks, translation_matrix=ttmat, attention_mask=masks, token_type_ids=segment_ids)
        
        # build CLS representation
        cls_tokens = result.pooler_output # first token
        cls_reps = []
        for i in range(cls_tokens.shape[0] // batch):
            cls_reps.append(cls_tokens[i*batch:(i+1)*batch])
        cls_reps = torch.stack(cls_reps, dim=2).mean(dim=2)
        
        return self.cls(cls_reps)

class CustomTransformerRankerPlaceboHead(TransformerRanker):
        
    def forward(self, query_tok, query_mask, doc_tok, doc_mask, query_subword_index, doc_subword_index, query_words, doc_words, tt, tt_threshold, norm, device):
        isPlacebo = True
        batch, ttmat, toks, masks, segment_ids, _ = self.custom_encode(query_tok, query_mask, doc_tok, doc_mask, query_subword_index, doc_subword_index, query_words, doc_words, tt, tt_threshold, isPlacebo, norm, device)
        # execute Transformer model
        if self.model_type == 'xlmroberta':
            result = self.transformer(input_ids=toks, attention_mask=masks)
        else:
            result = self.transformer(input_ids=toks, translation_matrix=ttmat, attention_mask=masks, token_type_ids=segment_ids)
        
        # build CLS representation
        cls_tokens = result.pooler_output # first token
        cls_reps = []
        for i in range(cls_tokens.shape[0] // batch):
            cls_reps.append(cls_tokens[i*batch:(i+1)*batch])
        cls_reps = torch.stack(cls_reps, dim=2).mean(dim=2)
        
        return self.cls(cls_reps)
