# ETF2Vec
Adapted the Natural Language Processing (NLP) concept of Word2Vec to 'dimensionalize' individual stock holdings from all Blackrock's US Public Equity ETF holdings into a 128 dimension 'ETF stock selection criteria' vector. This essentially treats tickers as individual words and finds context via their close-by weight % holding across various different ETF portfolios. Skipgrams were developed by restricting per ETF portfolio (similar to sentences). Negative selection sampling was done assuming a Zipf distribution.

My hope is to use this embedding data after significant compute time to further research into both active and passive ETF management.

skills: _keras_ functional API, _pandas_, Word2Vec

[Code](https://github.com/ryanjameskim/public/blob/master/210422%20ETF2Vec%20Keras%20Implementation.py)
