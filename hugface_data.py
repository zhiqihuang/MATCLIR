import random, json, os
from tqdm import tqdm
import torch
import mmap
from torch.utils.data import Dataset, DataLoader

# reads the number of lines in a file
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

# returns query and documents
def read_datafiles(files):
    queries = {}
    docs = {}
    for file_path in files:
        with open(file_path) as file:
            for line in tqdm(file, total=get_num_lines(file_path), desc='loading datafile (by line)'):
                cols = line.rstrip().split('\t')
                if len(cols) != 4:
                    tqdm.write(f'skipping line: `{line.rstrip()}`')
                    continue
                c_type, c_id, c_text, c_lang = cols
                assert c_type in ('query', 'doc')
                if c_type == 'query':
                    queries[c_id] = (c_text, c_lang)
                if c_type == 'doc':
                    docs[c_id] = (c_text, c_lang)
    return queries, docs

# returns qrels
def read_qrels_dict(file_path):
    result = {}
    with open(file_path) as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc='loading qrels (by line)'):
            qid, _, docid, score = line.split()
            result.setdefault(qid, {})[docid] = int(score)
    return result

# returns validation pairs
def read_run_dict(file_path, topK=10**5):
    result = {}
    with open(file_path) as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc='loading run (by line)'):
            qid, _, docid, rank, score, _ = line.split()
            if int(rank) <= topK:
                result.setdefault(qid, {})[docid] = float(score)
    return result

# returns training pairs
def read_pairs_dict(file_path):
    result = {}
    with open(file_path) as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc='loading pairs (by line)'):
            qid, docid = line.split()
            result.setdefault(qid, {})[docid] = 1
    return result

# returns predefined batches
def read_batches(file_path):
    with open(file_path) as file:
        result = json.load(file)
    return {int(k): v for k, v in result.items()}

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

class PredefinedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batches, batch_size):
        self.batch_size = batch_size
        self.shuffle = self.generate_batch_index(batches)
        self.epoch = -1

    def generate_batch_index(self, batches):
        shuffle = {}
        for epoch in batches:
            shuffle[epoch] = batches[epoch]['indices']
        return shuffle

    def __iter__(self):
        return iter(self.shuffle[self.epoch])

    def __len__(self):
        return len(self.shuffle[self.epoch])

################## vanilla train dataset ######################

class VanillaTrainCollator(object):
    def __init__(self, args):
        self.args = args
    
    def _pack_n_ship(self, items):
        QLEN = max(len(b) for b in items[1])
        MAX_DLEN = 800
        DLEN = min(MAX_DLEN, max(len(b) for b in items[3]))
        return {
            'query_id': items[0],
            'query_tok': self._pad_crop(items[1], QLEN),
            'doc_id': items[2],
            'doc_tok': self._pad_crop(items[3], DLEN),
            'query_mask': self._mask(items[1], QLEN),
            'doc_mask': self._mask(items[3], DLEN),
        }

    def _pad_crop(self, items, l):
        result = []
        for item in items:
            if len(item) < l:
                item = item + [-1] * (l - len(item))
            if len(item) > l:
                item = item[:l]
            result.append(item)
        return torch.tensor(result).long().to(self.args.device)

    def _mask(self, items, l):
        result = []
        for item in items:
            # needs padding (masked)
            if len(item) < l:
                mask = [1. for _ in item] + ([0.] * (l - len(item)))
            # no padding (possible crop)
            else:
                mask = [1. for _ in item[:l]]
            result.append(mask)
        return torch.tensor(result).float().to(self.args.device)
    
    def __call__(self, batch):
        qids = []
        dids = []
        qtoks = []
        dtoks = []
        for rec in batch:
            qids.append(rec['pos'][0])
            qids.append(rec['neg'][0])
            qtoks.append(rec['pos'][1])
            qtoks.append(rec['neg'][1])
            dids.append(rec['pos'][2])
            dids.append(rec['neg'][2])
            dtoks.append(rec['pos'][3])
            dtoks.append(rec['neg'][3])
        data = [qids, qtoks, dids, dtoks]
        return self._pack_n_ship(data)

class VanillaTrainDataset(Dataset):
    def __init__(self, args, model, queries, docs, train_pairs, qrels, batches):
        self.args = args
        self.qids = list(train_pairs.keys())
        self.queries = queries
        self.docs = docs
        self.tokenizer = model.tokenize
        self.train_pairs = train_pairs
        self.qrels = qrels
        self.all_comb = []
        self.batches = batches
        self.epoch = -1
        self.preprocessing()

    def preprocessing(self):
        self.qtoks = dict()
        self.dtoks = dict()
        self.pos_lookup = dict()
        self.neg_lookup = dict()
        #self.pos_pairs = list()
        for qid in tqdm(self.qids, desc="preload train data"):

            pos_ids = [did for did in self.train_pairs[qid] if self.qrels.get(qid, {}).get(did, 0) > 0 and did in self.docs]
            if len(pos_ids) == 0:
                continue
            self.pos_lookup[qid] = pos_ids
            for pos_id in pos_ids:
                if pos_id not in self.dtoks:
                    pos_doc_txt, pos_doc_lang = self.docs.get(pos_id)
                    pos_doc_tok = self.tokenizer(pos_doc_txt, pos_doc_lang)
                    self.dtoks[pos_id] = pos_doc_tok

            pos_ids_lookup = set(pos_ids)
            neg_ids = [did for did in self.train_pairs[qid] if did not in pos_ids_lookup and did in self.docs]
            if len(neg_ids) == 0:
                continue
            self.neg_lookup[qid] = neg_ids
            for neg_id in neg_ids:
                if neg_id not in self.dtoks:
                    neg_doc_txt, neg_doc_lang = self.docs.get(neg_id)
                    neg_doc_tok = self.tokenizer(neg_doc_txt, neg_doc_lang)
                    self.dtoks[neg_id] = neg_doc_tok
            
            qtxt, qlang = self.queries[qid]
            query_tok = self.tokenizer(qtxt, qlang)
            self.qtoks[qid] = query_tok
            #self.pos_pairs += [(qid, pos_id) for pos_id in pos_ids]
        self.qids = list(set(self.pos_lookup.keys()).intersection(set(self.neg_lookup.keys())))
        self.qids = sorted(self.qids)
        self.qid_index_dict = {i:qid for i, qid in enumerate(self.qids)}

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        query_tok = self.qtoks[qid]
        if self.args.sampler == "predefined":
            pos_id = self.batches[self.epoch]["pairs"][qid]['pos_id']
            neg_id = self.batches[self.epoch]["pairs"][qid]['neg_id']
        else:
            pos_id = random.choice(self.pos_lookup[qid])
            neg_id = random.choice(self.neg_lookup[qid])
        pos_doc_tok = self.dtoks[pos_id]
        neg_doc_tok = self.dtoks[neg_id]
        return {'pos': [qid, query_tok, pos_id, pos_doc_tok],
                'neg': [qid, query_tok, neg_id, neg_doc_tok]}

    # def __len__(self):
    #     return len(self.pos_pairs)

    # def __getitem__(self, item):
    #     qid, pos_id = self.pos_pairs[item]
    #     query_tok = self.qtoks[qid]
    #     if self.args.sampler == "predefined":
    #         neg_id = self.batches[self.epoch]["pairs"][qid][pos_id]
    #     else:
    #         neg_id = random.choice(self.neg_lookup[qid])
    #     pos_doc_tok = self.dtoks[pos_id]
    #     neg_doc_tok = self.dtoks[neg_id]
    #     return {'pos': [qid, query_tok, pos_id, pos_doc_tok],
    #             'neg': [qid, query_tok, neg_id, neg_doc_tok]}

def create_vanilla_train_loader(args, model, queries, docs, train_pairs, qrels, batches, batch_size):
    assert args.sampler in ["predefined", "random"]
    vanilla_traincollator = VanillaTrainCollator(args)
    cached_features_file = os.path.join(
        args.input_dir,
        "cached_train_vanilla_{}_{}_f{}".format(
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.rerank_topK,
            args.fold_num,
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("loading dataset from cached file")
        dataset = torch.load(cached_features_file)
    else:
        print("creating dataset")
        dataset = VanillaTrainDataset(args, model, queries, docs, train_pairs, qrels, batches)
        print("saving dataset into cached file")
        torch.save(dataset, cached_features_file)

    print("number of train query is ", dataset.__len__())
    if args.sampler == "predefined":
        datasampler = PredefinedBatchSampler(batches, batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=datasampler, shuffle=False, collate_fn=vanilla_traincollator)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=vanilla_traincollator)

################## custom train dataset ######################

class CustomTrainCollator(object):
    def __init__(self, args):
        self.args = args
    
    def _pack_n_ship(self, items):
        QLEN = max(len(b) for b in items[1]) # include all query tokens
        MAX_DLEN = 800
        DLEN = min(MAX_DLEN, max(len(b) for b in items[5]))
        query_sub_index = [w[:QLEN] for w in items[2]]
        query_words_txt = [w[:max(query_sub_index[i])] for i, w in enumerate(items[3])]
        doc_sub_index = [w[:MAX_DLEN] for w in items[6]]
        doc_words_txt = [w[:max(doc_sub_index[i])] for i, w in enumerate(items[7])]
        return {
            'query_id': items[0],
            'query_tok': self._pad_crop(items[1], QLEN),
            'query_sub_index': self._pad_crop_subs(items[2], QLEN),
            'query_words_txt': query_words_txt,
            'doc_id': items[4],
            'doc_tok': self._pad_crop(items[5], DLEN),
            'doc_sub_index': self._pad_crop_subs(items[6], DLEN),
            'doc_words_txt': doc_words_txt,
            'query_mask': self._mask(items[1], QLEN),
            'doc_mask': self._mask(items[5], DLEN),
            'max_len': QLEN + DLEN + 3,
        }

    def _pad_crop_subs(self, items, l):
        result = []
        for item in items:
            if len(item) < l:
                pad_list = list(range(max(item)+1, max(item)+ 1 + (l - len(item))))
                item = item + pad_list
            if len(item) > l:
                item = item[:l]
            result.append(item)
        return torch.tensor(result).long().to(self.args.device)

    def _pad_crop(self, items, l):
        result = []
        for item in items:
            if len(item) < l:
                item = item + [-1] * (l - len(item))
            if len(item) > l:
                item = item[:l]
            result.append(item)
        return torch.tensor(result).long().to(self.args.device)

    def _mask(self, items, l):
        result = []
        for item in items:
            # needs padding (masked)
            if len(item) < l:
                mask = [1. for _ in item] + ([0.] * (l - len(item)))
            # no padding (possible crop)
            else:
                mask = [1. for _ in item[:l]]
            result.append(mask)
        return torch.tensor(result).float().to(self.args.device)
    
    def __call__(self, batch):
        qids = []
        qsubs = []
        qwords = []
        dids = []
        dsubs = []
        dwords = []
        qtoks = []
        dtoks = []
        for rec in batch:
            qids.append(rec['pos'][0])
            qids.append(rec['neg'][0])
            qtoks.append(rec['pos'][1])
            qtoks.append(rec['neg'][1])
            qsubs.append(rec['pos'][2])
            qsubs.append(rec['neg'][2])
            qwords.append(rec['pos'][3])
            qwords.append(rec['neg'][3])
            dids.append(rec['pos'][4])
            dids.append(rec['neg'][4])
            dtoks.append(rec['pos'][5])
            dtoks.append(rec['neg'][5])
            dsubs.append(rec['pos'][6])
            dsubs.append(rec['neg'][6])
            dwords.append(rec['pos'][7])
            dwords.append(rec['neg'][7])
        data = [qids, qtoks, qsubs, qwords, dids, dtoks, dsubs, dwords]
        return self._pack_n_ship(data)

class CustomTrainDataset(Dataset):
    def __init__(self, args, model, queries, docs, train_pairs, qrels, batches):
        self.args = args
        self.qids = list(train_pairs.keys())
        self.queries = queries
        self.docs = docs
        self.tokenizer = model.custom_tokenize
        self.train_pairs = train_pairs
        self.qrels = qrels
        self.all_comb = []
        self.batches = batches
        self.epoch = -1
        self.preprocessing()

    def preprocessing(self):
        self.qtoks = dict()
        self.qsubs = dict()
        self.qwords = dict()
        self.dtoks = dict()
        self.dsubs = dict()
        self.dwords = dict()
        self.pos_lookup = dict()
        self.neg_lookup = dict()
        # self.pos_pairs = list()
        for qid in tqdm(self.qids, desc="preload train data"):

            # pick a positive document randomly for qid
            pos_ids = [did for did in self.train_pairs[qid] if self.qrels.get(qid, {}).get(did, 0) > 0 and did in self.docs]
            if len(pos_ids) == 0:
                continue
            self.pos_lookup[qid] = pos_ids
            for pos_id in pos_ids:
                if pos_id not in self.dtoks:
                    pos_doc_txt, pos_doc_lang = self.docs.get(pos_id)
                    pos_doc_tok, pos_sub_index, pos_words_txt = self.tokenizer(pos_doc_txt, pos_doc_lang)
                    self.dtoks[pos_id] = pos_doc_tok
                    self.dsubs[pos_id] = pos_sub_index
                    self.dwords[pos_id] = pos_words_txt

            # pick a negative document randomly for qid (non-positive ones are negative)
            pos_ids_lookup = set(pos_ids)
            neg_ids = [did for did in self.train_pairs[qid] if did not in pos_ids_lookup and did in self.docs]
            if len(neg_ids) == 0:
                continue
            self.neg_lookup[qid] = neg_ids
            for neg_id in neg_ids:
                if neg_id not in self.dtoks:
                    neg_doc_txt, neg_doc_lang = self.docs.get(neg_id)
                    neg_doc_tok, neg_sub_index, neg_words_txt = self.tokenizer(neg_doc_txt, neg_doc_lang)
                    self.dtoks[neg_id] = neg_doc_tok
                    self.dsubs[neg_id] = neg_sub_index
                    self.dwords[neg_id] = neg_words_txt
            qtxt, qlang = self.queries[qid]
            query_tok, query_sub_index, query_words_txt = self.tokenizer(qtxt, qlang)
            self.qtoks[qid] = query_tok
            self.qsubs[qid] = query_sub_index
            self.qwords[qid] = query_words_txt
            # self.pos_pairs += [(qid, pos_id) for pos_id in pos_ids]
        self.qids = list(set(self.pos_lookup.keys()).intersection(set(self.neg_lookup.keys())))
        self.qids = sorted(self.qids)
        self.qid_index_dict = {i:qid for i, qid in enumerate(self.qids)}
        
    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        query_tok = self.qtoks[qid]
        query_sub_index = self.qsubs[qid]
        query_words_txt = self.qwords[qid]
        if self.args.sampler == "predefined":
            pos_id = self.batches[self.epoch]["pairs"][qid]['pos_id']
            neg_id = self.batches[self.epoch]["pairs"][qid]['neg_id']
        else:
            pos_id = random.choice(self.pos_lookup[qid])
            neg_id = random.choice(self.neg_lookup[qid])
        pos_doc_tok = self.dtoks[pos_id]
        pos_sub_index = self.dsubs[pos_id]
        pos_words_txt = self.dwords[pos_id]
        neg_doc_tok = self.dtoks[neg_id]
        neg_sub_index = self.dsubs[neg_id]
        neg_words_txt = self.dwords[neg_id]
        return {'pos': [qid, query_tok, query_sub_index, query_words_txt, pos_id, pos_doc_tok, pos_sub_index, pos_words_txt],
                'neg': [qid, query_tok, query_sub_index, query_words_txt, neg_id, neg_doc_tok, neg_sub_index, neg_words_txt]}

    # def __len__(self):
    #     return len(self.pos_pairs)

    # def __getitem__(self, item):
    #     qid, pos_id = self.pos_pairs[item]
    #     query_tok = self.qtoks[qid]
    #     query_sub_index = self.qsubs[qid]
    #     query_words_txt = self.qwords[qid]
    #     if self.args.sampler == "predefined":
    #         neg_id = self.batches[self.epoch]["pairs"][qid][pos_id]
    #     else:
    #         neg_id = random.choice(self.neg_lookup[qid])
    #     pos_doc_tok = self.dtoks[pos_id]
    #     pos_sub_index = self.dsubs[pos_id]
    #     pos_words_txt = self.dwords[pos_id]
    #     neg_doc_tok = self.dtoks[neg_id]
    #     neg_sub_index = self.dsubs[neg_id]
    #     neg_words_txt = self.dwords[neg_id]
    #     return {'pos': [qid, query_tok, query_sub_index, query_words_txt, pos_id, pos_doc_tok, pos_sub_index, pos_words_txt],
    #             'neg': [qid, query_tok, query_sub_index, query_words_txt, neg_id, neg_doc_tok, neg_sub_index, neg_words_txt]}

def create_custom_train_loader(args, model, queries, docs, train_pairs, qrels, batches, batch_size):
    assert args.sampler in ["predefined", "random"]
    custom_traincollator = CustomTrainCollator(args)
    cached_features_file = os.path.join(
        args.input_dir,
        "cached_train_custom_{}_{}_f{}".format(
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.rerank_topK,
            args.fold_num,
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("loading dataset from cached file")
        dataset = torch.load(cached_features_file)
    else:
        print("creating dataset")
        dataset = CustomTrainDataset(args, model, queries, docs, train_pairs, qrels, batches)
        print("saving dataset into cached file")
        torch.save(dataset, cached_features_file)
    print("number of train query is ", dataset.__len__())
    if args.sampler == "predefined":
        datasampler = PredefinedBatchSampler(batches, batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=datasampler, shuffle=False, collate_fn=custom_traincollator)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_traincollator)


################## vanilla run dataset ######################
class VanillaRunCollator(object):
    def __init__(self, args):
        self.args = args
    
    def _pack_n_ship(self, items):
        QLEN = max(len(b) for b in items[1])
        MAX_DLEN = 800
        DLEN = min(MAX_DLEN, max(len(b) for b in items[3]))
        return {
            'query_id': items[0],
            'query_tok': self._pad_crop(items[1], QLEN),
            'doc_id': items[2],
            'doc_tok': self._pad_crop(items[3], DLEN),
            'query_mask': self._mask(items[1], QLEN),
            'doc_mask': self._mask(items[3], DLEN),
        }

    def _pad_crop(self, items, l):
        result = []
        for item in items:
            if len(item) < l:
                item = item + [-1] * (l - len(item))
            if len(item) > l:
                item = item[:l]
            result.append(item)
        return torch.tensor(result).long().to(self.args.device)

    def _mask(self, items, l):
        result = []
        for item in items:
            # needs padding (masked)
            if len(item) < l:
                mask = [1. for _ in item] + ([0.] * (l - len(item)))
            # no padding (possible crop)
            else:
                mask = [1. for _ in item[:l]]
            result.append(mask)
        return torch.tensor(result).float().to(self.args.device)
    
    def __call__(self, batch):
        qids = []
        dids = []
        qtoks = []
        dtoks = []
        for rec in batch:
            qids.append(rec[0])
            qtoks.append(rec[1])
            dids.append(rec[2])
            dtoks.append(rec[3])
        data = [qids, qtoks, dids, dtoks]
        return self._pack_n_ship(data)
        
class VanillaRunDataset(Dataset):
    def __init__(self, args, model, queries, docs, run, name):
        self.args = args
        self.qids = list(run.keys())
        self.queries = queries
        self.docs = docs
        self.tokenizer = model.tokenize
        self.run = run
        self.name = name
        self.preprocessing()
    
    def preprocessing(self):
        self.data = []
        self.length = 0
        for qid in tqdm(self.qids, desc="preload {} data".format(self.name)):
            qtxt, qlang = self.queries[qid]
            query_tok = self.tokenizer(qtxt, qlang)
            for did in self.run[qid]:
                doc_txt, doc_lang = self.docs.get(did)
                if doc_txt is None:
                    continue
                doc_tok = self.tokenizer(doc_txt, doc_lang)
                self.data.append([qid, query_tok, did, doc_tok])
                self.length += 1
        self.data = sorted(self.data, key=lambda x: len(x[3]))

    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        return self.data[item]

def create_vanilla_run_loader(args, model, queries, docs, run, name, batch_size):
    vanilla_runcollator = VanillaRunCollator(args)
    cached_features_file = os.path.join(
        args.input_dir,
        "cached_{}_vanilla_{}_{}_f{}".format(
            name,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.rerank_topK,
            args.fold_num,
        ),
    )
    if os.path.exists(cached_features_file)and not args.overwrite_cache:
        print("loading dataset from cached file")
        dataset = torch.load(cached_features_file)
    else:
        print("creating dataset")
        dataset = VanillaRunDataset(args, model, queries, docs, run, name)
        print("saving dataset into cached file")
        torch.save(dataset, cached_features_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=vanilla_runcollator)

################## custom run dataset ######################
class CustomRunCollator(object):
    def __init__(self, args):
        self.args = args
    
    def _pack_n_ship(self, items):
        QLEN = max(len(b) for b in items[1])
        MAX_DLEN = 800
        DLEN = min(MAX_DLEN, max(len(b) for b in items[5]))
        query_sub_index = [w[:QLEN] for w in items[2]]
        query_words_txt = [w[:max(query_sub_index[i])] for i, w in enumerate(items[3])]
        doc_sub_index = [w[:MAX_DLEN] for w in items[6]]
        doc_words_txt = [w[:max(doc_sub_index[i])] for i, w in enumerate(items[7])]
        return {
            'query_id': items[0],
            'query_tok': self._pad_crop(items[1], QLEN),
            'query_sub_index': self._pad_crop_subs(items[2], QLEN),
            'query_words_txt': query_words_txt,
            'doc_id': items[4],
            'doc_tok': self._pad_crop(items[5], DLEN),
            'doc_sub_index': self._pad_crop_subs(items[6], DLEN),
            'doc_words_txt': doc_words_txt,
            'query_mask': self._mask(items[1], QLEN),
            'doc_mask': self._mask(items[5], DLEN),
            'max_len': QLEN + DLEN + 3
        }

    def _pad_crop_subs(self, items, l):
        result = []
        for item in items:
            if len(item) < l:
                pad_list = list(range(max(item)+1, max(item)+ 1 + (l - len(item))))
                item = item + pad_list
            if len(item) > l:
                item = item[:l]
            result.append(item)
        return torch.tensor(result).long().to(self.args.device)

    def _pad_crop(self, items, l):
        result = []
        for item in items:
            if len(item) < l:
                item = item + [-1] * (l - len(item))
            if len(item) > l:
                item = item[:l]
            result.append(item)
        return torch.tensor(result).long().to(self.args.device)

    def _mask(self, items, l):
        result = []
        for item in items:
            # needs padding (masked)
            if len(item) < l:
                mask = [1. for _ in item] + ([0.] * (l - len(item)))
            # no padding (possible crop)
            else:
                mask = [1. for _ in item[:l]]
            result.append(mask)
        return torch.tensor(result).float().to(self.args.device)
    
    def __call__(self, batch):
        qids = []
        qsubs = []
        qwords = []
        dids = []
        dsubs = []
        dwords = []
        qtoks = []
        dtoks = []
        for rec in batch:
            qids.append(rec[0])
            qtoks.append(rec[1])
            qsubs.append(rec[2])
            qwords.append(rec[3])
            dids.append(rec[4])
            dtoks.append(rec[5])
            dsubs.append(rec[6])
            dwords.append(rec[7])
        data = [qids, qtoks, qsubs, qwords, dids, dtoks, dsubs, dwords]
        return self._pack_n_ship(data)
        
class CustomRunDataset(Dataset):
    def __init__(self, args, model, queries, docs, run, name):
        self.args = args
        self.qids = list(run.keys())
        self.queries = queries
        self.docs = docs
        self.tokenizer = model.custom_tokenize
        self.run = run
        self.name = name
        self.preprocessing()
    
    def preprocessing(self):
        self.data = []
        self.length = 0
        for qid in tqdm(self.qids, desc="preload {} data".format(self.name)):
            qtxt, qlang = self.queries[qid]
            query_tok, query_sub_index, query_words_txt = self.tokenizer(qtxt, qlang)
            for did in self.run[qid]:
                doc_txt, doc_lang = self.docs.get(did)
                if doc_txt is None:
                    continue
                doc_tok, doc_sub_index, doc_words_txt = self.tokenizer(doc_txt, doc_lang)
                self.data.append([qid, query_tok, query_sub_index, query_words_txt, did, doc_tok, doc_sub_index, doc_words_txt])
                self.length += 1
        self.data = sorted(self.data, key=lambda x: len(x[5]))

    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        return self.data[item]

def create_custom_run_loader(args, model, queries, docs, run, name, batch_size):
    custom_runcollator = CustomRunCollator(args)
    cached_features_file = os.path.join(
        args.input_dir,
        "cached_{}_custom_{}_{}_f{}".format(
            name,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.rerank_topK,
            args.fold_num
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("loading dataset from cached file")
        dataset = torch.load(cached_features_file)
    else:
        print("creating dataset")
        dataset = CustomRunDataset(args, model, queries, docs, run, name)
        print("saving dataset into cached file")
        torch.save(dataset, cached_features_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_runcollator)

#############################################################################################################################
def iter_train_pairs(args, model, queries, docs, train_pairs, qrels, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_train_pairs(model, queries, docs, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(args, batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}

def _iter_train_pairs(model, ds_queries, ds_docs, train_pairs, qrels):
    while True:
        qids = list(train_pairs.keys())
        random.shuffle(qids)
        for qid in qids:
            qtxt, qlang = ds_queries[qid]
            query_tok = model.tokenize(qtxt, qlang)
            # pick a positive document randomly for qid
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                continue
            pos_id = random.choice(pos_ids)
            pos_doc_txt, pos_doc_lang = ds_docs.get(pos_id)
            if pos_doc_txt is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            pos_doc_tok = model.tokenize(pos_doc_txt, pos_doc_lang)

            # pick a negative document randomly for qid (non-positive ones are negative)
            pos_ids_lookup = set(pos_ids)
            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
            if len(neg_ids) == 0:
                continue
            neg_id = random.choice(neg_ids)
            neg_doc_txt, neg_doc_lang = ds_docs.get(neg_id)
            if neg_doc_txt is None:
                tqdm.write(f'missing doc {neg_id}! Skipping')
                continue
            neg_doc_tok = model.tokenize(neg_doc_txt, neg_doc_lang)

            yield qid, pos_id, query_tok, pos_doc_tok
            yield qid, neg_id, query_tok, neg_doc_tok


def iter_valid_records(args, model, queries, docs, run, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_valid_records(model, queries, docs, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(args, batch)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    # final batch
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(args, batch)


def _iter_valid_records(model, ds_queries, ds_docs, run):
    for qid in run:
        qtxt, qlang = ds_queries[qid]
        query_tok = model.tokenize(qtxt, qlang)
        for did in run[qid]:
            doc_txt, doc_lang = ds_docs.get(did)
            if doc_txt is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok = model.tokenize(doc_txt, doc_lang)
            yield qid, did, query_tok, doc_tok


def _pack_n_ship(args, batch):
    QLEN = 20
    MAX_DLEN = 800
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(args, batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(args, batch['doc_tok'], DLEN),
        'query_mask': _mask(args, batch['query_tok'], QLEN),
        'doc_mask': _mask(args, batch['doc_tok'], DLEN),
    }


def _pad_crop(args, items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result).long().to(args.device)


def _mask(args, items, l):
    result = []
    for item in items:
        # needs padding (masked)
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        # no padding (possible crop)
        else:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result).float().to(args.device)
