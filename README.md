For detailed instructions, please reference "instructions" file.

This package will perform the following sequential steps:

1. Gather and clean raw 10-K filings from SEC online database
2. Preprocess each document
3. Cluster each company into industry groupings
4. Display the results of the NMF cluster model
5. Perform sentiment analysis at the industry level
6. Perform sentiment analysis at the company level

Example code to implement a NMF clustering model for 2018 where the number of clusters equals 15:

# import package
import Russell2000_NMF

# acquire data
Russell2000_NMF.data_acquisition.get(0,100,2018)

# run the model
results = Russell2000_NMF.nmf.nmf(2018,15)

# display results
Russell2000_NMF.nmf.display(results)

# perform industry sentiment analysis
industry_sentiment = Russell2000_NMF.nmf.industry_sentiment(2018,15,results)

# perform company level sentiment analysis
company_sentiment = Russell2000_NMF.nmf.company_sentiment(2018,15,results)
