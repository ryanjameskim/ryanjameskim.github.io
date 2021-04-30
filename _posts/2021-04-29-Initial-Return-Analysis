# Initial ETF Return and STD analysis

Why do we want to represent ETF stock selection as a vector in the first place?

* We can numerically compare two stocks using cosine similarity to group certain kinds of stocks together in non-obvious ways. For instance, in a [previous post](https://ryanjameskim.com/2021/04/27/ETF2Vec-Optimized.html), we showed that Apple clustered near other big cap technology names like AMZN and GOOG, but also was similar to non-obvious names like Berkshire Hathaway, likely because of its large market cap.

* We can represent ETF portfolios (or any collection of stocks) by weighting each of the underlying stock vectors by their proportion inside the portfolio, coming up with a vector of the same dimensions that represents the ETF.

* From this, we can compare ETFs to underlying stocks and calculate return and standard deviation data.

For instance, perhaps we wanted to compare the YTD return performance of SOXX, the iShares PHLX Semiconductor ETF against a basket of 15 stocks closest to the weighted sum of its components. Maybe the idea is we would rather buy 15 of the closest stocks to get some benefits of diversification but desire increased concentration.

The SOXX weighted vector is closest to these 15 stocks:
Top Stock 1     AVGO
Top Stock 2     NVDA
Top Stock 3     INTC
Top Stock 4     AMAT
Top Stock 5      TXN
Top Stock 6     ALLE
Top Stock 7      SLM
Top Stock 8     SCHW
Top Stock 9      FHN
Top Stock 10    CTRN
Top Stock 11    AMCR
Top Stock 12    VRSK
Top Stock 13     PFE
Top Stock 14     TGT
Top Stock 15     RGA

The equal weight performance of this 15 stock portfolio outperformed the SOXX return by nearly 13% (26.6% versus 14.0%) and the portfolio's standard deviation of returns was more than 1% lower than the ETF's (2.4% versus 1.3%).

Here is a list of all such ETF's in which the blended portfolio of similar stocks had a lower standard deviatin than the ETF itself, but achieved a higher cumulative return.

                                                   Name ETF CumReturn Blended Stock CumReturn   ETF STD Blended STD
SOXX                     iShares-PHLX-Semiconductor-ETF      0.139678                0.266399  0.024573    0.013023
ICLN                    iShares-Global-Clean-Energy-ETF     -0.148725                0.499032  0.030706    0.019736
IEO   iShares-U.S.-Oil-&-Gas-Exploration-&-Productio...      0.410274                0.426485  0.025161     0.01726
IGV           iShares-Expanded-Tech-Software-Sector-ETF      0.044733                0.145203   0.01685     0.01085
IWFH      iShares-Virtual-Work-and-Life-Multisector-ETF      0.010734                0.139716  0.019277    0.014233
MTUM               iShares-MSCI-USA-Momentum-Factor-ETF       0.08161                0.102994  0.016975     0.01196
TECB             U.S.-Tech-Breakthrough-Multisector-ETF      0.082388                0.110707  0.014192    0.010804
IYW                         iShares-U.S.-Technology-ETF      0.108647                0.198057  0.015649    0.012341
BFTR                    BlackRock-Future-Innovators-ETF      0.138986                0.196323  0.023018    0.020065
IMCG             iShares-Morningstar-Mid-Cap-Growth-ETF      0.058395                0.212409   0.01509    0.012318
IDU                          iShares-U.S.-Utilities-ETF      0.053934                0.164463  0.010201    0.007562
BTEK                          BlackRock-Future-Tech-ETF      0.044848                0.315401  0.021723     0.01944
IYK                     iShares-U.S.-Consumer-Goods-ETF      0.045422                0.185047  0.011516    0.009369
IDNA     iShares-Genomics-Immunology-and-Healthcare-ETF      0.058119                0.154329  0.022998    0.020862
IXN                             iShares-Global-Tech-ETF      0.091585                0.175144  0.014582    0.012546
IGM                    iShares-Expanded-Tech-Sector-ETF      0.109298                0.186876  0.014683     0.01286
SDG                      iShares-MSCI-Global-Impact-ETF      0.048685                0.085232  0.009589    0.007883
IHF               iShares-U.S.-Healthcare-Providers-ETF      0.117616                0.178413  0.010387    0.008933
IETC                iShares-Evolved-U.S.-Technology-ETF      0.105751                0.190261  0.014224    0.012977
IWP                  iShares-Russell-Mid-Cap-Growth-ETF      0.067387                0.438096   0.01498    0.013786
JXI                        iShares-Global-Utilities-ETF      0.035459                0.162881  0.008842    0.007933
IXP                    iShares-Global-Comm-Services-ETF      0.138758                0.174328  0.011622    0.010776
RXI           iShares-Global-Consumer-Discretionary-ETF       0.09766                0.105311  0.011728    0.010917
IXG                       iShares-Global-Financials-ETF      0.180799                0.226169  0.010587    0.010189
MXI                        iShares-Global-Materials-ETF      0.142611                0.153375  0.011846    0.011449
IDRV               iShares-Self-Driving-EV-and-Tech-ETF      0.106481                0.223945  0.014842    0.014519
IHAK                 iShares-Cybersecurity-and-Tech-ETF      0.026766                0.131815  0.015618    0.015349
IEHS        iShares-Evolved-U.S.-Healthcare-Staples-ETF      0.102585                0.204981  0.009017    0.008755
IGF                   iShares-Global-Infrastructure-ETF      0.062915                0.160254   0.00864    0.008413

[Code](https://github.com/ryanjameskim/public/blob/master/210428%20ETF%20v%20Vector%20Return%20comparison.py)

