##ETF2Vec Midpoint Summary

Originally posted to the NYC Python Discord Chat.

#ETF2VEC#
hi there guys, i'm sorry i can't make it tonight, but i wanted to share a small project i've been working on that applies Word2Vec to ETF stock selection. Basically, the project vectorizes a universe of stocks from Blackrock's ETF holdings and tries to condense portfolio inclusion information as a vector. For instance, some ETFs pick stocks based on metrics like 'value' and others on 'size' and this vector is a way to possibly summarize all these selection categories into a 128-dimensional number. I used negative sampling and trained similar to how Word2Vec creates word embeddings, imagining stock tickers like 'words' and etf portfolios as 'sentences'. You can see beginning to end code and some early visualizations at my website, links below.

[Scrapper](https://ryanjameskim.com/2021/03/31/BlackrockETFDownloader.html)
[Holding Filter](https://ryanjameskim.com/2021/04/21/DBFilter.html)
[XLS data cleaner](https://ryanjameskim.com/2021/04/20/DataCleaner.html)
[ETF2Vec training (SGD)](https://ryanjameskim.com/2021/04/22/ETF2Vec-Keras.html)
[ETF2Vec batch training](https://ryanjameskim.com/2021/04/27/ETF2Vec-Optimized.html) --has some cool visualizations and a link to tensorflow's embeddings projector with the data

Next, I'm going to work on using return data and daily variance to see if there are better ways to construct an ETF with less underling stocks or vice-versa, if a portfolio of single names is better expressed as an ETF. I'd ideally like to host this as a quick app on my website, so might need some help from the python webdevs!

Please feel free to reach out with any comments or suggestions.
