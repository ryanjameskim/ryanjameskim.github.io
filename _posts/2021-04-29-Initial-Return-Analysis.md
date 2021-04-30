# Initial ETF Return and STD analysis

[Code](https://github.com/ryanjameskim/public/blob/master/210428%20ETF%20v%20Vector%20Return%20comparison.py)

Why do we want to represent ETF stock selection as a vector in the first place?

* We can numerically compare two stocks using cosine similarity to group certain kinds of stocks together in non-obvious ways. For instance, in a [previous post](https://ryanjameskim.com/2021/04/27/ETF2Vec-Optimized.html), we showed that Apple clustered near other big cap technology names like AMZN and GOOG, but also was similar to non-obvious names like Berkshire Hathaway, likely because of its large market cap.

* We can represent ETF portfolios (or any collection of stocks) by weighting each of the underlying stock vectors by their proportion inside the portfolio, coming up with a vector of the same dimensions that represents the ETF.

* From this, we can compare ETFs to underlying stocks and calculate return and standard deviation data.

For instance, perhaps we wanted to compare the YTD return performance of SOXX, the iShares PHLX Semiconductor ETF against a basket of 15 stocks closest to the weighted sum of its components. Maybe the idea is we would rather buy 15 of the closest stocks to get some benefits of diversification but desire increased concentration.

The SOXX weighted vector is closest to these 15 stocks: (1) AVGO, (2) NVDA, (3) INTC, (4) AMAT, (5) TXN, (6) ALLE, (7) SLM, (8) SCHW, (9) FHN, (10) CTRN, (11) AMCR, (12) VRSK, (13) PFE, (14) TGT, (15)  RGA. As you can see, while there are many semiconductor names in the list, there are other outliers like TGT, Target, as well.

The equal weight performance of this 15 stock portfolio outperformed the SOXX return by nearly 13% (26.6% versus 14.0%) and the portfolio's standard deviation of returns was more than 1% lower than the ETF's (2.4% versus 1.3%).

Here is a list of all such ETF's in which the blended portfolio of similar stocks had a lower standard deviatin than the ETF itself, but achieved a higher cumulative return.

|      	| Name                                                	| ETF CumReturn 	| Blended Stock CumReturn 	| ETF STD 	| Blended STD 	|
|------	|-----------------------------------------------------	|---------------	|-------------------------	|---------	|-------------	|
| SOXX 	| iShares-PHLX-Semiconductor-ETF                      	| 13.97         	| 26.64                   	| 2.46    	| 1.3         	|
| ICLN 	| iShares-Global-Clean-Energy-ETF                     	| -14.87        	| 49.9                    	| 3.07    	| 1.97        	|
| IEO  	| iShares-U.S.-Oil-&-Gas-Exploration-&-Production-ETF 	| 41.03         	| 42.65                   	| 2.52    	| 1.73        	|
| IGV  	| iShares-Expanded-Tech-Software-Sector-ETF           	| 4.47          	| 14.52                   	| 1.68    	| 1.09        	|
| IWFH 	| iShares-Virtual-Work-and-Life-Multisector-ETF       	| 1.07          	| 13.97                   	| 1.93    	| 1.42        	|
| MTUM 	| iShares-MSCI-USA-Momentum-Factor-ETF                	| 8.16          	| 10.3                    	| 1.7     	| 1.2         	|
| TECB 	| U.S.-Tech-Breakthrough-Multisector-ETF              	| 8.24          	| 11.07                   	| 1.42    	| 1.08        	|
| IYW  	| iShares-U.S.-Technology-ETF                         	| 10.86         	| 19.81                   	| 1.56    	| 1.23        	|
| BFTR 	| BlackRock-Future-Innovators-ETF                     	| 13.9          	| 19.63                   	| 2.3     	| 2.01        	|
| IMCG 	| iShares-Morningstar-Mid-Cap-Growth-ETF              	| 5.84          	| 21.24                   	| 1.51    	| 1.23        	|
| IDU  	| iShares-U.S.-Utilities-ETF                          	| 5.39          	| 16.45                   	| 1.02    	| 0.76        	|
| BTEK 	| BlackRock-Future-Tech-ETF                           	| 4.48          	| 31.54                   	| 2.17    	| 1.94        	|
| IYK  	| iShares-U.S.-Consumer-Goods-ETF                     	| 4.54          	| 18.5                    	| 1.15    	| 0.94        	|
| IDNA 	| iShares-Genomics-Immunology-and-Healthcare-ETF      	| 5.81          	| 15.43                   	| 2.3     	| 2.09        	|
| IXN  	| iShares-Global-Tech-ETF                             	| 9.16          	| 17.51                   	| 1.46    	| 1.25        	|
| IGM  	| iShares-Expanded-Tech-Sector-ETF                    	| 10.93         	| 18.69                   	| 1.47    	| 1.29        	|
| SDG  	| iShares-MSCI-Global-Impact-ETF                      	| 4.87          	| 8.52                    	| 0.96    	| 0.79        	|
| IHF  	| iShares-U.S.-Healthcare-Providers-ETF               	| 11.76         	| 17.84                   	| 1.04    	| 0.89        	|
| IETC 	| iShares-Evolved-U.S.-Technology-ETF                 	| 10.58         	| 19.03                   	| 1.42    	| 1.3         	|
| IWP  	| iShares-Russell-Mid-Cap-Growth-ETF                  	| 6.74          	| 43.81                   	| 1.5     	| 1.38        	|
| JXI  	| iShares-Global-Utilities-ETF                        	| 3.55          	| 16.29                   	| 0.88    	| 0.79        	|
| IXP  	| iShares-Global-Comm-Services-ETF                    	| 13.88         	| 17.43                   	| 1.16    	| 1.08        	|
| RXI  	| iShares-Global-Consumer-Discretionary-ETF           	| 9.77          	| 10.53                   	| 1.17    	| 1.09        	|
| IXG  	| iShares-Global-Financials-ETF                       	| 18.08         	| 22.62                   	| 1.06    	| 1.02        	|
| MXI  	| iShares-Global-Materials-ETF                        	| 14.26         	| 15.34                   	| 1.18    	| 1.14        	|
| IDRV 	| iShares-Self-Driving-EV-and-Tech-ETF                	| 10.65         	| 22.39                   	| 1.48    	| 1.45        	|
| IHAK 	| iShares-Cybersecurity-and-Tech-ETF                  	| 2.68          	| 13.18                   	| 1.56    	| 1.53        	|
| IEHS 	| iShares-Evolved-U.S.-Healthcare-Staples-ETF         	| 10.26         	| 20.5                    	| 0.9     	| 0.88        	|
| IGF  	| iShares-Global-Infrastructure-ETF                   	| 6.29          	| 16.03                   	| 0.86    	| 0.84        	|

