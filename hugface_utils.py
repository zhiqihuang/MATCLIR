import math
import torch

def subwords_index(subwords):
    index = 0
    wordpieces = []
    for sw in subwords:
        if sw[:2] != '##':
            index +=1
        wordpieces.append(index)
    return wordpieces

def subbatch(toks, maxlen):
    _, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    S = math.ceil(DLEN / SUBBATCH) if SUBBATCH > 0 else 0 # minimize the size given the number of subbatch
    if SUBBATCH == 1:
        return toks, SUBBATCH
    else:
        stack = []
        for s in range(SUBBATCH):
            stack.append(toks[:, s*S:(s+1)*S]) 
            if stack[-1].shape[1] != S:
                nulls = torch.zeros_like(toks[:, :S - stack[-1].shape[1]])
                stack[-1] = torch.cat([stack[-1], nulls], dim=1)
        return torch.cat(stack, dim=0), SUBBATCH

def subbatch_sub_index(toks, maxlen):
    _, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    S = math.ceil(DLEN / SUBBATCH) if SUBBATCH > 0 else 0 # minimize the size given the number of subbatch
    if SUBBATCH == 1:
        return toks, SUBBATCH
    else:
        stack = []
        for s in range(SUBBATCH):
            stack.append(toks[:, s*S:(s+1)*S]) 
            if stack[-1].shape[1] != S:
                nulls = []
                for chunk in stack[-1]:
                    start = chunk[-1]+1
                    end = start + S - len(chunk)
                    nulls.append(torch.arange(start, end).long().to(toks.device))
                nulls = torch.stack(nulls, dim=0)
                stack[-1] = torch.cat([stack[-1], nulls], dim=1)
        return torch.cat(stack, dim=0), SUBBATCH


def un_subbatch(embed, toks, maxlen):
    BATCH, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    if SUBBATCH == 1:
        return embed
    else:
        embed_stack = []
        for b in range(SUBBATCH):
            embed_stack.append(embed[b*BATCH:(b+1)*BATCH])
        embed = torch.cat(embed_stack, dim=1)
        embed = embed[:, :DLEN]
        return embed

def getTransTableDict(dir_tt):
    translation_tb = dict()
    with open(dir_tt, "r") as trfile:
        for row in trfile:
            tokens = row.split(" ")
            source = tokens[0]
            target = tokens[1]
            prob = float(tokens[2])
            if source not in translation_tb:
                translation_tb[source] = dict()
            translation_tb[source][target] = prob
    return translation_tb

def query_translator_subwords(tt, q_subwords_index, q_words, d_words, maxd, qd_subwords_index, threshold, stopwords, is_placebo=False, reduction="norm"):
    ttmat = []
    for q_index, q_words_clean, d_words_clean, qd_index in zip(q_subwords_index, q_words, d_words, qd_subwords_index):
        qd_index_dict = {}
        for i, pos in enumerate(qd_index):
            if pos in qd_index_dict:
                qd_index_dict[pos].append(i)
            else:
                qd_index_dict[pos] = [i]

        q_words_clean_dict = {}
        for i, w in enumerate(q_words_clean):
            if w == '' or w == 'unk':
                continue
            if w in q_words_clean_dict:
                q_words_clean_dict[w] += qd_index_dict[i+1].copy()
            else:
                q_words_clean_dict[w] = qd_index_dict[i+1].copy()
        
        d_words_clean_dict = {}
        offset = max(q_index.tolist()) + 2
        for i, w in enumerate(d_words_clean):
            if w == '' or w == 'unk':
                continue
            if w in d_words_clean_dict:
                d_words_clean_dict[w] += qd_index_dict[i+offset].copy()
            else:
                d_words_clean_dict[w] = qd_index_dict[i+offset].copy()

        wmat = torch.eye(maxd, requires_grad=False)
        
        if not is_placebo:
            for i, pos in qd_index_dict.items():
                for p in pos:
                    wmat[p][pos] = 1.0
        
            for qw, qlocs in q_words_clean_dict.items():
                if qw in stopwords['query']:
                    continue
                # translations
                trans = tt.get(qw, {})
                if qw not in trans:
                    trans.update({qw:1.0})
                for t in trans:
                    #if trans[t] > threshold:
                    if t not in stopwords['document'] and trans[t] > threshold:
                        dlocs = d_words_clean_dict.get(t, [])
                        if dlocs != []:
                            for qloc in qlocs:
                                wmat[qloc][dlocs] = trans[t]
                            for dloc in dlocs:
                                wmat[dloc][qlocs] = trans[t]
        
        if reduction == 'norm':
            wmat = torch.div(wmat, wmat.sum(dim=1, keepdim=True))
        if reduction == 'softmax':
            wmat = torch.nn.functional.softmax(wmat, dim=1)
        wmat = wmat.unsqueeze(0)
        ttmat.append(wmat)
    ttmat = torch.cat(ttmat, dim=0)
    return ttmat