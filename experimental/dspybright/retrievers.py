import bm25s
import Stemmer
import numpy as np

class BM25S:
    def __init__(self, data, doc_key='doc', lang='en'):
        assert lang == 'en'
        
        self.doc_key   = doc_key
        self.n_docs    = len(data)
        self.docs      = [xx[doc_key] for xx in data]
        self.meta      = [{k:v for k,v in xx.items() if k != doc_key} for xx in data]
        self.ids       = np.array([xx['id'] for xx in data])

        self.stemmer   = Stemmer.Stemmer('english')
        self.retriever = bm25s.BM25(k1=0.9, b=0.4, method='lucene', backend='numba', weight_query=True)
        self.retriever.index(bm25s.tokenize(self.docs, stopwords="en", stemmer=self.stemmer))
    
    def run(self, query, return_docs=False, topk=100, excluded_ids=None):
        assert isinstance(query, str), "BM25S: Batching is disallowed!"
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)

        if excluded_ids is None:
            idxs, scores = self.retriever.retrieve(query_tokens, k=topk)
        else:
            weight_mask = np.isin(self.ids, excluded_ids, invert=True)
            assert weight_mask.sum() == self.n_docs - len(excluded_ids), f'{weight_mask.sum()} != {self.n_docs - len(excluded_ids)}'
            idxs, scores = self.retriever.retrieve(query_tokens, k=topk, weight_mask=weight_mask)
        
        idxs, scores = idxs[0], scores[0]
        
        out = []
        for idx, score in zip(idxs, scores):
            out.append({
                **({self.doc_key : self.docs[idx]} if return_docs else {}),
                **(self.meta[idx]),
                "_score" : float(score),
            })
        
        return out
