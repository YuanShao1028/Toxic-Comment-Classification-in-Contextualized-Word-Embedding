# Toxic Comment Classification in Contextualized Word Embedding

The goal of this challenge is to classify negative comments from Kaggle Jigsaw dataset which is mainly composed of wikipedia human labeled comments into six possible emotion categories including toxic, severe toxic, obscene, threat, insult and identity hate. Generally, pre-trained word embedding such as word2vec typically powers the deeping learning model with initialization of only the lowest layer in pretrained word vectors weights. The contextualized word representation models provide more robust embedding which are useful for transfer learning in natural language processing. In this challenge, we designed several deep model architectures with a range of contextualized word representations for multi-label classification and performed evaluations on benchmark dataset.
<ol>
<li>GloVe</li>
    We adopt Glove as baseline word embedding in this project. Glove is glove vectors for word representation obtained via unsupervised learning.  Glove model is similar to Word2vec in its assumption and modeling except that the context words is not only the limited neighboring words but the whole corpus. Specifically, aggregated global word-word co-occurrence statistics from a corpus is used in training process to generate word embedding.

<li>CoVe</li>
	 The main idea of Cove is to transfer knowledge from an encoder pretrained on machine learning translation to a variety of downstream natural language processing task. The deep LSTM encoder from an attentional sequence-to-sequence model trained for machine translation is used to contextualize word vectors. Specifically, context vectors(CoVe) is extracted from pretrained LSTM which acts as an encoder for machine translation.

<li>Elmo</li>
	Elmo is a type of deep contextualized word representation, modeling complex characteristics of word use in various linguistic contexts. All layers of a deep pre-trained bidirectional language model (biLM) trained on a large text corpus are combined to generate contextualized word representation. The representation for each word is not a fixed vector,  but depends on the entire context in which it leverages. Further, ELMo representations are based on characters, enabling the network to utilize morphological clues to generate robust representations for out-of-vocabulary tokens unseen in training process.

<li>GenSen</li>
	GenSen is designed to learn general purpose, fixed-length representations of sentences via multi-task training. The model aims to combine the inductive biases of several diverse training signals used to learn sentence representations into a single model. Specifically, the multi-task framework includes a combination of sequence-to-sequence tasks such as multilingual NMT, constituency parsing and skip-thought vectors as well as a classification task - natural language inference.  

<li>Sent2Vec</li>
	Sent2Vec generates robust sentence representation useful for transfer learning by jointly learning from multiple text classification tasks and combining them with pre-trained word-level and sentence level encoders. Further, adversarial training enables private encoders to learn only task-specific features and the shared encoder to learn generic features.
</ol>

This repo refers to codes and model from  https://github.com/wasiahmad/transferable_sent2vec
