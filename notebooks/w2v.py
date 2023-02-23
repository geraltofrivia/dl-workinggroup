from collections import Counter
from tqdm.auto import tqdm, trange
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load the dataset
wikitext = load_dataset("wikitext", "wikitext-2-v1")
wikitext, wikitext['train'][10]['text']

# Make a vocab
word_counter = Counter()
for line in tqdm(wikitext['train'], miniters=1000):
    word_counter.update(line['text'].split())

# len(word_counter), word_counter.most_common(10)

unk_token = '<unk>'
vocab = {tok: i for i, (tok, freq) in enumerate(word_counter.most_common())}
n_words = len(vocab)

# Tokenize the corpus
train_text = [doc['text'].split() for doc in wikitext['train']]


# # Emulate the model here
# inputs = torch.randint(0,1000, (10, 2))
# labels_pos = torch.randint(0, 2, (10,))
# labels_neg = torch.randint(0, 2, (10,))
#
# U = torch.nn.Embedding(1000, 20)
# V = torch.nn.Embedding(1000, 20)
#
# vec_u = U(inputs[:,0])
# vec_vp = V(inputs[:,1])
# vec_vn = V(inputs[:,1])
#
# # Lets do the positive dot

class SkipGramWordEmbeddings(torch.nn.Module):

  def __init__(self, vocab_size, emb_dimension):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb_dimension = emb_dimension
    self.U = torch.nn.Embedding(vocab_size, emb_dimension)
    self.V = torch.nn.Embedding(vocab_size, emb_dimension)

    initrange = 1.0 / self.emb_dimension
    torch.nn.init.uniform_(self.U.weight.data, -initrange, initrange)
    self.V.weight.data.uniform_(-initrange, initrange)
#     torch.nn.init.constant_(self.V.weight.data, 0)

  def forward(self, u, pos, neg):
    vec_u = self.U(u)  # (bs, 300)
    vec_pos_v = self.V(pos) # (bs, 300)
    vec_neg_v = self.V(neg) # (bs, 300)

    score = torch.mul(vec_u, vec_pos_v)
    score = torch.sum(score, dim=1)
    score = - F.logsigmoid(score)

    neg_score = torch.bmm(vec_neg_v, vec_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    neg_score = - F.logsigmoid(-1*neg_score).squeeze()
    
    loss = score + neg_score

    return loss.mean()


class W2VIter:

    def __init__(self, vocab, corpus, negatives=4, batchsize=64):
        """
            vocab: dict: key is token, value is id
            corpus: List of [ List of tokens ]
            batchsize: int
        """

        # Count Word Frequency
        wfreq = Counter()
        for doc in corpus:
            wfreq.update(doc)

        # Shuffle the corpus
        npr = np.random.permutation(len(corpus))
        corpus = [corpus[i] for i in npr]

        self._batchsize = batchsize - batchsize % (
                    negatives + 1)  # rounded off to negatives+1. E.g. if bs 63, and neg=4; bs = 60
        self._negatives = negatives
        self._vocab = vocab
        self._wordfreq = {tok: wfreq[tok] for tok, _ in vocab.items()}
        self._unkid = self._vocab[unk_token]

        # Convert corpus to wordids
        corpus = [self.get_word_ids(doc) for doc in corpus]

        # Convert every document to word pairs (shuffled)
        wordpairs = [self.get_word_pairs(doc) for doc in tqdm(corpus) if doc]

        self.data = [x for x in wordpairs if x]

    def get_word_ids(self, doc):
        return [self._vocab.get(tok, self._unkid) for tok in doc]

    def get_word_pairs(self, doc):
        pairs = []
        for i, token in enumerate(doc):
            if i - 1 >= 0 and doc[i - 1] != self._unkid:
                pairs.append((doc[i - 1], token))
            if i - 2 >= 0 and doc[i - 2] != self._unkid:
                pairs.append((doc[i - 2], token))
            if i + 1 < len(doc) and doc[i + 1] != self._unkid:
                pairs.append((doc[i + 1], token))
            if i + 2 < len(doc) and doc[i + 2] != self._unkid:
                pairs.append((doc[i + 2], token))

        # Shuffle the pairs
        npr = np.random.permutation(len(pairs))
        pairs = [pairs[i] for i in npr]
        return pairs

    def __iter__(self):
        self.docid, self.wordid = 0, 0
        return self

    def __next__(self):
        bs = int(self._batchsize / (self._negatives + 1))
        batch_pos = []

        while True:

            # If we have already gone through all the documents
            if self.docid == len(self.data):
                raise StopIteration  # the loop stops,the epoch is over

            # get next document
            document = self.data[self.docid]

            # upto: either batchsize, or doc length whichever is shorter (if doc has 100 pairs, take 60) (if batchsize is 60)
            _from = self.wordid
            _upto = _from + int(min(bs, len(document) - _from))
            batch_pos += document[_from:_upto]

            # What to do with global pointers
            if _upto >= len(document):
                # Lets move to the next document
                self.docid += 1
                self.wordid = 0
            else:
                # Still in the same document
                self.wordid = _upto

            # If the batch is over i.e. we got as many pairs as we wanted, we break this while loop
            if len(batch_pos) == int(self._batchsize / (self._negatives + 1)):
                break
            # If not, we still continue taking pairs from the next document
            else:
                bs -= (_upto - _from)
                
        batch_pos = torch.tensor(batch_pos)
        u = batch_pos[:,0]
        v_pos = batch_pos[:,1]
        
        # Negatives: for one positive there would be multiple negatives
        v_neg = torch.randint(0, len(self._vocab), (v_pos.shape[0], self._negatives))

        return u, v_pos, v_neg  # warning they have different shapes
    
# Okay I guess its time to trainnnn
model = SkipGramWordEmbeddings(len(vocab), 300)
if torch.cuda.is_available():
    model.cuda()
    
    
dataiter = W2VIter(vocab, corpus=train_text, negatives=4, batchsize=1000)

epochs = 10
lr = 0.2
opt = torch.optim.SGD(model.parameters(), lr=lr)


model.train()

per_epoch_loss = []
for e in trange(epochs):
    
    per_batch_loss = []
    for u, pos, neg in tqdm(dataiter):
        
        if torch.cuda.is_available():
            u = u.cuda()
            pos = pos.cuda()
            neg = neg.cuda()
        
        # reset gradients
        opt.zero_grad()
        
        loss = model(u, pos, neg)
        loss.backward()
        opt.step()
        
        per_batch_loss.append(loss.cpu().detach().item())
    
    per_epoch_loss.append(per_batch_loss)
    print(f"{e:4d}: Loss = {np.mean(per_batch_loss)}")