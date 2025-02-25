# Natural Language Process


### Dependency Parser

#### Nueral Dependency Parser

## Language Modeling

### N-Gram Models
1. Issues with using frequency estimate to predict the next word?
   
   new sentences are created all the time, and we won't be able to count all the new sentences beforehand.

1. Hidden Markov Assumption for BiGram
   
   The assumption that the probability of a word depends only on the previous word is called a Markov assumption:
   $$
        P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-1})
   $$

1. MLE of the entire sequence
   
   $$
    P(w_{1:n}) \approx \prod_{k=1}^{n} P(w_k \mid w_{k-1})
   $$

1. General case of n-gram parameter estimation
   
   $$
    P(w_n \mid w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n-1} \ w_n)}{C(w_{n-N+1:n-1})}
   $$


### Perplexity
1. What is Perperlexity (PPL) ?
   
   The perplexity (sometimes abbreviated as PP or PPL) of a language model on a test set is the inverse probability of the test set (one over the probability of the test set), normalized by the number of words.

1. The formulation of perplexity.
   $$
    \text{perplexity}(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i)}}
    $$
    where:
    - $N$ is the total number of words in the sentence
    - $w_i$ is the i-th word in the sentence
1. What is smoothing ?
   
   To keep a language model from assigning zero probability to these unseen events, we’ll have to shave off a bit of probability mass from some more frequent events and give it to the events we’ve never seen. This modification is called **smoothing** or **discounting**

2. Laplacian Smoothing Formulation
   
   Add one to all the n-gram counts, before we normalize them into probabilities

   $$
    P_{\text{Laplace}}(w | C) = \frac{\text{count}(w, C) + \alpha}{\sum_{w'} \text{count}(w', C) + \alpha V}
    $$

    where:
    - $C$ is the total nunber of word $w$'s count
    - $w$ is the the word $w$
    - $\alpha$ is the hyper parameter
    - $V$ is the vocab size


### Recurrent Neural Networks

### Neural Machine Translation

### Natural Language Generation





