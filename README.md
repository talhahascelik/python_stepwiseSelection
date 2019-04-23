
This script is about an automated stepwise backward and forward feature selection. You can easily apply on Dataframes.
Functions returns not only the final features but also elimination iterations, so you can track what exactly happend at the iterations.

You can apply it on both Linear and Logistic problems. Eliminations can be apply with Akaike information criterion (AIC), Bayesian information criterion (BIC), R-squared (Only works with linear), Adjusted R-squared (Only works with linear)

Enjoy the code!


Required Libraries: pandas, numpy, statmodels

See more about stepwise regression : https://en.wikipedia.org/wiki/Stepwise_regression
