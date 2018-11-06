# Toxic Comment Classification in Contextualized Word Embedding

The goal of this challenge is to classify negative comments from Kaggle Jigsaw dataset which is mainly composed of wikipedia human labeled comments into six possible emotion categories including toxic, severe toxic, obscene, threat, insult and identity hate. Generally, pre-trained word embedding such as word2vec typically powers the deeping learning model with initialization of only the lowest layer in pretrained word vectors weights. The contextualized word representation models provide more robust embedding which are useful for transfer learning in natural language processing. In this challenge, we designed several deep model architectures with a range of contextualized word representations for multi-label classification and performed evaluations on benchmark dataset.

This repo refers to codes and model from  https://github.com/wasiahmad/transferable_sent2vec
