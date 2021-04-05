# Covid-Analysis

## Used Data

- Temperature data: [TerraClimate Dataset](http://www.climatologylab.org/terraclimate.html)
- COVID-19 data: [JHU CSSE COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

## Goal

The goal of the project was to verify the following hypotheses:

* Temperature has a significant impact on COVID-19 contagiousness:
  * The test was performed by calculating the mean temperature per country and month and comparing it to the normalized reproduction coefficient per country and month.
  The hypothesis was verified using ANOVA test.
  
* There is a significant difference between the death ratio in different european countries:
  The hypothesis was tested using two different methods:
  * By comparison of number of deaths and number of comfirmed cases using chi2 test
  * By comparison of death ratio per month in different countries using ANOVA test
 
## Requirements
* Python 3.7
* See `requirements.txt`
