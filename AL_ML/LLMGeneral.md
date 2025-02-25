# Questions on Large Language Models

## Transformers
### What is Q K V?

### Why division by $\sqrt{d*k}$
Normalizing the distribution of softmax, preventing gradient vanishing.
### Are there other ways to achieve similar functions to self-attention?
### Are there any other way than divided by $\sqrt{d*k}$?
Any methods that could keep the weights within the inverval is good. See Google T5 (good normalization).
### They reason why transformers use Layer Norm
Preventing vanishing gradient.
### Why does not use Batch Norm
- Sequence length of NLP tasks are not the same across each batch
- hard to scale batch sizes since Transformers are usually very large
  
### Why multi-head attention
- It could enable different heads to focus on different parts when handling sentences

### Positional Encoding


### Different types of attention

## Tokenizer
### Byte Pair Encoding
### Sentence Piece

# Model Training
## Pretraining
### Norm
#### RMS Norm
Formulation:
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum{x_i^2 + \epsilon}}}
$$

- layer norm center the data while RMS norm does not center the data (normalizing using the mean value across feature), thus it is more efficient.
- RMS also preserve the original information of the data since only scale the data.

#### Post Deepnorm



### Warm up
## Finetuning
### Difference between LoRA and SFT
### LoRA






### Deepspeed


