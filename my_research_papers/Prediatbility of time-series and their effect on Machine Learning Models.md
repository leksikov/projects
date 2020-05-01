
# Predictability of time-series and their effect on Machine Learning Models

## Abstract
We represent the market daily returns as weighted sums of returns with different period.

Based on this Logistic regression was constructed. In order to determine the efficacy of Momentum based model with simple benchmark we generated synthetic dataset.

Using synthetic dataset the controlled experiment was conducted. The results of controlled. From controlled experiment we can conclude that :

1.  Momentum model is effective for trending and mean-reversal time-series
    
2.  Simple Logistic Regression gives highest possible accuracy. It works similarly as for trending and for mean-reversal. While it doesn’t work for random.
    

Based on this experiment we also explored the hurst exponent and determined that hurst and accuracy are correlated. Hence based on hurst exponent we can know how much time-series is predictable and how much the accuracy we can expect from ML model.

## Applications
-   Application 1. Use Hurst as Stock Screener
    

	-   We want to make sure that we do not trade Random Walk series
	    
	-   If the stock exhibit persistence we want to apply tending strategies
	    
	-   If stock is anti-persistent then mean-reverting strategies are applied
	    

-   Application 2. Filter to improve accuracy of ML model
	    

	-   ML makes prediction for every available data point. And it always makes binary prediction in any case. But what if time-series in random mode temporarily?
	    
	-   Moving average of Hurst or similar indicators can be used and the threshold is set to distinguish Non-random movements where we make a predictions
    

-   Application 3. Risk Management for existing quant strategies    

	-   Suppose we have Moving Average Crossover strategy
	    
	-   Since it is trending strategy it will provide best performance when time-series is in persistent mode.
	    
	-   We can detect persistent mode using moving average of Hurst with determined specific threshold

## Conclusion
-   Momentum Model was proposed to model market movements and to use it as basis for Machine Learning model
    
-   It was shown that momentum model can work under trending and mean-reverting regime
    
-   Synthetic dataset usage was introduced to conduct the controlled experiments in order to determine how quantitative model and machine learning model are robust under different market conditions
    
-   Predictive power of the models under different market conditions was measured and compared
    
-   It was established as long as Hurst measurements deviates from random walk value the higher accuracy of machine learning model
    
-   3 possible practical applications were suggested about how to apply time-series predictability measures
    
-   Use case with accuracy improvement was demonstrated for the ML model

## Copyrighted&copy;2020 Sergey Leksikov
No copy and usage of ideas is allowed in any form