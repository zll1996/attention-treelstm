# Neural_Sentiment_Analysis
Implementation of Tree Structured LSTM and Attention Mechanism Models for the task of Sentiment Analysis on Stanford Sentiment Treebank

In this project we have implemented following models:

1. Linear LSTM model (baseline)
2. Tree Structured LSTM model taking reference from Kai Sheng Tai's paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075).
3. Tree Structure LSTM with Attention Mechanism.


### Requirements
- [PyTorch](http://pytorch.org/) Deep learning library for the implementaion of Neural Models
- [tqdm](https://github.com/tqdm/tqdm): display progress bar
- Java >= 8 (for Stanford CoreNLP utilities i.e. Stanford Parsers)
- Python >= 3


References:

1. Code for baseline has been referenced from https://github.com/adeshpande3/LSTM-Sentiment-Analysis
2. COde for Tree LSTM has been referenced from https://github.com/ttpro1995/TreeLSTMSentiment/
