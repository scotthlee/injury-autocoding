# Autocoding Injury Narratives with BERT
This repo contains code for training an ensemble of BERT models to autocode injury narratives. 

## BERT
BERT stands for Bidirectional Encoder Representations from Transformers. One of the newer large-scale contextual language models, it's a good baseline for a wide variety of downstream NLP tasks. To learn more about how the base model is trained, check out the paper on [arXiv](https://arxiv.org/abs/1810.04805). To see how folks from Google implemented it in TensorFlow, check out the original [repo]() on GitHub, which we've also included here (but not updated in while).

## Data
To get the data, including the small base BERT model we fine-tuned to classify the narratives, download [this .zip file](https://www.dropbox.com/s/10iu4rslh6pre81/injury_autocoding.zip?dl=1). Once you've upzipped the file, you'll see a directory with a BERT folder, two CSV files with information about the injury codes, and a few empty folders for holding the individual model checkpoints that go into our final ensemble.
