[![Substack][substack-shield]][substack-url]
[![Nitter][nitter-shield]][nitter-url]
[![Mastodon][mastodon-shield]][mastodon-url]
[![Linkedin][linkedin-shield]][linkedin-url]

# Corrected Text Classification with CNNs

## Text Classification with CNNs in PyTorch 

[Fernando Lopez][linkedin-fernando] ([@FernandoLpz](https://github.com/FernandoLpz)) attempted to implement the architecture proposed in the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf).
He even drew up some nice diagrams in his Mediam blog post [Text Classification with CNNs in PyTorch](https://medium.com/@fer.neutron/text-classification-with-cnns-in-pytorch-1113df31e79f?sk=12e7c4b3092297ee0e1c71d659297043).
I don't use Medium because of their exploitative and deceptive business practices.
So I couldn't comment on his blog post to explain the error in his implementation.

Fernando represented the sentences (sequences of words) as sequences of integers.
This is common in PyTorch implementations, but the integers are never used as the input to a Deep Learning model because the numerical values of word indices do not contain any information about the words themselves other than their position in a one-hot vector.
The index values are arbitrary and destroy all the information contained in a sentence.
The numerical values of word indices contain no information about the meaning of the words. Word indices are fundamentally a categorical variable, not an ordinal or numerical value.
Fernando's code effectively randomizes the 1-D numerical representation of a word thus the  sequence of indices passed into the CNN is meaningless.

This can be seen in the poor accuracy on the test set compared to the training set. 
The test set accuracy is little better than random guessing or always guessing the majority class.
In pyTorch a word embeddings layers is typically utilized to reconstruct a meaningful vector.

This repository attempts to correct this fundamental error by replacing word indices with progressively more informative numerical representations of words:

- removing the CNN entirely
- 2-D CNN on one-hot vectors
- 2-D CNN on TF-IDF vectors
- 2-D CNN on LSA vectors (PCA on TF-IDF vectors)
- 2-D CNN on word vectors

In addition, a baseline LinearModel (a single-layer perceptron, or fully-connected feed-forward layer) is trained on the average of the word vectors for a sentence or tweet (`spacy.Doc.vector`).

## Links


[substack-shield]: https://proai.substack.com
[substack-url]: https://proai.substack.com

[mastodon-shield]: https://img.shields.io/mastodon/follow/@hobson?style=social
[mastodon-url]: https://mastodon.social/users/hobson

[nitter-url]: https://nitter.net/hobsonlane
[nitter-shield]: https://nitter.net/hobsonlane

[linkedin-shield]: https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://www.linkedin.com/in/hobsonlane/

[linkedin-fernando]: https://www.linkedin.com/in/fernando-lopezvelasco/