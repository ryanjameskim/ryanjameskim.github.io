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

|      	| Name                                                	| ETF CumReturn        	| Blended Stock CumReturn 	| ETF STD              	| Blended STD          	|
|------	|-----------------------------------------------------	|----------------------	|-------------------------	|----------------------	|----------------------	|
| SOXX 	| iShares-PHLX-Semiconductor-ETF                      	| 0.13967777045434643  	| 0.2663990031387093      	| 0.024573102384879572 	| 0.013022724131690117 	|
| ICLN 	| iShares-Global-Clean-Energy-ETF                     	| -0.14872517314558467 	| 0.4990318055990564      	| 0.030706482316768112 	| 0.01973643154748195  	|
| IEO  	| iShares-U.S.-Oil-&-Gas-Exploration-&-Production-ETF 	| 0.41027440887468536  	| 0.4264851134529281      	| 0.025161113283210076 	| 0.017259928027991027 	|
| IGV  	| iShares-Expanded-Tech-Software-Sector-ETF           	| 0.04473311512245512  	| 0.14520314163659342     	| 0.016849614711615342 	| 0.010850347338483817 	|
| IWFH 	| iShares-Virtual-Work-and-Life-Multisector-ETF       	| 0.010734385368625237 	| 0.13971634823250126     	| 0.01927704767285534  	| 0.014232710448819456 	|
| MTUM 	| iShares-MSCI-USA-Momentum-Factor-ETF                	| 0.08160993647428846  	| 0.10299427346063358     	| 0.016975140807360078 	| 0.011959860437818166 	|
| TECB 	| U.S.-Tech-Breakthrough-Multisector-ETF              	| 0.0823883894582771   	| 0.11070654064265287     	| 0.014192467609471814 	| 0.01080415651883018  	|
| IYW  	| iShares-U.S.-Technology-ETF                         	| 0.10864713242590723  	| 0.19805676407308917     	| 0.01564912109684088  	| 0.01234061050484038  	|
| BFTR 	| BlackRock-Future-Innovators-ETF                     	| 0.138986286909274    	| 0.1963234841387807      	| 0.02301773881685059  	| 0.02006529915286279  	|
| IMCG 	| iShares-Morningstar-Mid-Cap-Growth-ETF              	| 0.05839518195968815  	| 0.21240893946809655     	| 0.015090240239774844 	| 0.012318127297263752 	|
| IDU  	| iShares-U.S.-Utilities-ETF                          	| 0.05393440300192908  	| 0.16446293395140363     	| 0.010201039330653087 	| 0.007562400013597009 	|
| BTEK 	| BlackRock-Future-Tech-ETF                           	| 0.04484839744189605  	| 0.3154005995675841      	| 0.02172317908657917  	| 0.019440250916220102 	|
| IYK  	| iShares-U.S.-Consumer-Goods-ETF                     	| 0.0454222507945048   	| 0.18504725510919906     	| 0.011516008852002135 	| 0.009368952293647408 	|
| IDNA 	| iShares-Genomics-Immunology-and-Healthcare-ETF      	| 0.058119330941604985 	| 0.15432865642329022     	| 0.022997680218700637 	| 0.020861749287668558 	|
| IXN  	| iShares-Global-Tech-ETF                             	| 0.09158477621831812  	| 0.17514377959089683     	| 0.014582372337003328 	| 0.012545900212562988 	|
| IGM  	| iShares-Expanded-Tech-Sector-ETF                    	| 0.10929810020979787  	| 0.1868764222603974      	| 0.01468300292371399  	| 0.01286000956297616  	|
| SDG  	| iShares-MSCI-Global-Impact-ETF                      	| 0.048685105302476    	| 0.08523202449997004     	| 0.009588643259820697 	| 0.007883033049743039 	|
| IHF  	| iShares-U.S.-Healthcare-Providers-ETF               	| 0.11761596914119735  	| 0.17841317165071022     	| 0.010386641894274547 	| 0.008932941605342153 	|
| IETC 	| iShares-Evolved-U.S.-Technology-ETF                 	| 0.10575056326428013  	| 0.19026075536868756     	| 0.014223803446023975 	| 0.012977249933770398 	|
| IWP  	| iShares-Russell-Mid-Cap-Growth-ETF                  	| 0.06738730101883592  	| 0.4380960133505047      	| 0.01497991462687248  	| 0.013785837364794351 	|
| JXI  	| iShares-Global-Utilities-ETF                        	| 0.035458649557788505 	| 0.16288136338142029     	| 0.008841512296964822 	| 0.00793339221178485  	|
| IXP  	| iShares-Global-Comm-Services-ETF                    	| 0.13875784664640178  	| 0.17432774659408176     	| 0.011621823880319427 	| 0.010775701940423156 	|
| RXI  	| iShares-Global-Consumer-Discretionary-ETF           	| 0.0976597858883958   	| 0.1053107510844636      	| 0.01172785481067299  	| 0.0109173829702205   	|
| IXG  	| iShares-Global-Financials-ETF                       	| 0.1807988409882961   	| 0.2261694516372616      	| 0.010587214861069552 	| 0.010189233828001173 	|
| MXI  	| iShares-Global-Materials-ETF                        	| 0.1426113801584395   	| 0.15337515640961935     	| 0.011845819367584618 	| 0.011448879893260598 	|
| IDRV 	| iShares-Self-Driving-EV-and-Tech-ETF                	| 0.10648144427968052  	| 0.22394500089499936     	| 0.01484152157845321  	| 0.014518707960384    	|
| IHAK 	| iShares-Cybersecurity-and-Tech-ETF                  	| 0.026765845647983316 	| 0.13181470471550144     	| 0.015617775734105112 	| 0.01534933374229759  	|
| IEHS 	| iShares-Evolved-U.S.-Healthcare-Staples-ETF         	| 0.1025848861670672   	| 0.2049814907903496      	| 0.009017294610226352 	| 0.008755196370453909 	|
| IGF  	| iShares-Global-Infrastructure-ETF                   	| 0.06291466615420034  	| 0.16025425643599042     	| 0.008639849825245374 	| 0.008412665846275303 	|


[Code](https://github.com/ryanjameskim/public/blob/master/210428%20ETF%20v%20Vector%20Return%20comparison.py)

