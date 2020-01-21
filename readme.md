# Stock price prediction from news headline using RNN

Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices, company performance data and news data suitable for machine learning algorithms to process.
Can we actually predict stock prices with machine learning? Investors make educated guesses by analyzing data. They'll read the news, study the company history, industry trends and other lots of data points that go into making a prediction. The prevailing theories is that stock prices are totally random and unpredictable but that raises the question why top firms like Morgan Stanley and Citigroup hire quantitative analysts to build predictive models.
This project utilizes Deep Learning models, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict stock prices. For data with timeframes recurrent neural networks (RNNs) come in handy but recent researches have shown that LSTM networks are the most popular and useful variants of RNNs.
We have used Keras to build a LSTM to predict stock prices using historical stock prices and news data and visualize both the predicted price values over time and the optimal parameters for the model.

## Data Description
The dataset consists of:
Top 5 world news headlines from Kaggle & The opening, closing, adjusted closing, high and low price for 15 companies from Yahoo Finance for approximately eight years (from 08-08-2008 to 01-07-2016).The 15 American based multinational technology companies used in our data are as follows:

	1.Baidu Inc
	2.Adobe Systems Incorporated
	3.Oracle Corporation
	4.Amazon.com, Inc.
	5.Alphabet Inc Class C
	6.NVIDIA Corporation
	7.Microsoft Corporation
	8.NetEase Inc (ADR)
	9.Electronic Arts Inc.
	10.Apple Inc.
	11.QUALCOMM, Inc.
	12.Cisco Systems, Inc.
	13.Texas Instruments Incorporated
	14.Intel Corporation
	15.IBM Common Stock

The details and the result of the project can be found in Report.docx and Results.docx respectively.