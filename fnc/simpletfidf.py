# my tf-idf implementation
class SimpleTfidf():
    def __init__(self, tokenizer, stemmer, stopwords):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopwords = stopwords
        
    def fit(self, texts):
        tokens = self._extract_tokens(texts)
        vocab = list(self._extract_vocab(tokens))
        self.matrix = lil_matrix((len(vocab), len(tokens)), dtype=float)
        idfs = self._extract_idfs(vocab, tokens)
        self._set_tf_values(idfs, vocab, tokens)
        self.vocab = vocab
        
    def _extract_tokens(self, texts):
        tokens = []
        for text in texts:
            tt = list(filter(lambda x: x not in self.stopwords, [self.stemmer.stem(token) for token in self.tokenizer(text)]))
            tokens.append(tt)
        return tokens
        
    def _extract_vocab(self, tokens):
        return reduce((lambda vocab, t: set(list(vocab) + t)), tokens, set())
    
    def _extract_idfs(self, vocab, tokens):
        D = len(tokens)
        idfs = []
        for word in vocab:
            n = sum([1 if word in text else 0 for text in tokens])
            idf = D/n
            idfs.append(idf)
        return idfs
    
    
    def _set_tf_values(self, idfs, vocab, tokens):
        for i in range(len(tokens)):
            text = tokens[i]
            for j in range(len(idfs)):
                word = vocab[j]
                idf = idfs[j]
                N = len(text)
                n = sum([1 if w == word else 0 for w in text])
                tf = n/N
                tf_idf = tf*idf
                self.matrix[j, i] = tf_idf
