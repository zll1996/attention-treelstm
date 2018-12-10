# Neural_Sentiment_Analysis
Implementation of Tree Structured LSTM and Attention Mechanism Models for the task of Sentiment Analysis on Stanford Sentiment Treebank

In this project we have implemented following models:

1. Linear LSTM model (baseline)
2. Tree Structured LSTM model taking reference from Kai Sheng Tai's paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075).
3. Tree Structure LSTM with Attention Mechanism.


### Software Requirements
- [PyTorch](http://pytorch.org/) Deep learning library for the implementaion of Neural Models
- [Tensorflow](https://www.tensorflow.org/) Deep learning library by Google for the implementaion of Neural Models
- [tqdm](https://github.com/tqdm/tqdm): display progress bar
- Java >= 8 (for Stanford CoreNLP utilities i.e. Stanford Parsers)
- Python >= 3 for running the core system
- Python 2.7 for running preprocessing scripts

### Development and Testing Environment Used
- Operating Systems: macOS Mojave and Ubuntu 18.04
- Processor: Intel i5 Quad Core
- RAM: 8 GB DDR3

## Usage
First run the script `./fetch_and_preprocess.sh`

This downloads the following data:
  - [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) (sentiment classification task)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!

and the following libraries:

  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
  - [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)

Now to test the baseline model goto the `baseline` directory using `cd ./baseline` and run `python baseline.py`

### References:

1. Code for baseline has been referenced from https://github.com/adeshpande3/LSTM-Sentiment-Analysis
2. Code for Tree LSTM has been referenced from https://github.com/ttpro1995/TreeLSTMSentiment/

### License
Apache
