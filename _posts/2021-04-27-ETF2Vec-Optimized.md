# ETF2Vec Optimized

I reworked the ETF2Vec code to run much quicker in batches and also saved the cosine similarity test to the end. I used Tensorflow's Embedding visualization tool to look at the data. Below you can see the visualization for AAPL, Apple Inc.'s vector reduced down to three dimensions using PCA. Its nearest neighbors are highlighted and include names like Google (GOOG) and Amazon (AMZN), as might be expected.

![AAPL visualization]({{ site.baseurl }}/images/ETF2VecAAPLexample.PNG)

All 10 of AAPL's nearest neighbors (cosine).

![AAPL 10 nearest neighbors]({{ site.baseurl }}/images/ETF2VecAAPLexamplenearest.PNG)


skills: _keras_, embedding, Word2Vec, tensorflow embeddings projector, google colabs

[Jupyter Notebook](https://github.com/ryanjameskim/public/blob/master/210427%20ETF2Vec%20Batch%20Implementation.ipynb)

[Visualization](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/ryanjameskim/0e408ac0fac14a2a811a4979d22a3715/raw/2ad177885ee9109a74ce4e3163a289845559a4ea/etf2vecjson.json)
