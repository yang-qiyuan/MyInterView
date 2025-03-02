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

## Attention
  
### Why multi-head attention
- It could enable different heads to focus on different parts when handling sentences

### Flash Attention

### MLA


## Positional Encoding
[Link](https://0809zheng.github.io/2022/07/01/posencode.html)
#### What is the shortcomings of learnable PE?
You cannot exprapolate since the length of learnable parameters are typically fixed.
#### Sinusoidal Embeddings
![Sinusoidal embed](../pics/sinusoid_emb.png)
Formula:
![Sin Formula](../pics/sin_emed_formula.png)
Pros and Cons\
Pros:
- It is very bad at extrapolation. It may not have an ideal on how to deal with very long sequences.
- Sin and Cos functions are continuous, providing **smooth** information
- No additional trainable paprameters
- 
Cons:
- No relative information, not expressive enough
- Fixed frequency, not necessarily optimal for every NLP tasks.

#### RoPE
![rope](../pics/rope_formula.png)
Pros and Cons\
Pros:
- Encodes relative position naturally, depending on relative positions, making it easier to generalize.
- It is based on continuous rotation, so it could handle longer sequences.
- It is parameter efficient and cheap to compute
  
Cons:
- Does not encode absolute position, hard for tasks require absolute position
- Hard to interpret since it is complex
  



### Different types of attention

### Parameter Calculation (only include decoder block)
![transformer](../pics/transformer.png)
Let $d_h$ denote the hidden state size of the model, $d_{qkv}$ denote the dimension of the vector of $Q, K, V$, $N$ denote the size of heads. Then we have:

1. **The size of the Scaled Dot Product Attention** is: $W_Q + W_K + W_B + W_O$ = $(d_{h}*d_{qkv} + d_{qkv})*3h + d_{h}*d_{h} + d_{h}$ = $4*d_{h}^2$
   
2. **The size of the MLP** is: $d_{h}*d_{ffn} + d_{h} + d_{h}*d_{ffn} + d_{h}$

![layernorm](../pics/layernorm.png)
3. **The size of Layer Norm** is: $\beta, \gamma$ = $d_{h}$, which is $d_{h}*2$
4. **Output size** is: $d_{h}*d_{vocab}$


## Tokenizer
### Byte Pair Encoding
### Sentence Piece

# Model Training
## Pretraining
### Norm
#### RMS Norm
Formulation:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum{x_i^2 + \epsilon}}}$$

- layer norm center the data while RMS norm does not center the data (normalizing using the mean value across feature), thus it is more efficient.
- RMS also preserve the original information of the data since only scale the data.

#### Layer Norm
Pros: 
- Stablize training and convergence.
- Works well for sequence model since it does not. depends on batch sizes and normalizes across features.

Cons:
- Higher computation overhead.
- Adds up two more learnable parameters.
- Less effective for CNN.

#### Post Deepnorm



### Warm up

### ZeRO
#### ZeRO 1
#### ZeRO 2
#### ZeRO 3

## Finetuning
### Difference between LoRA and SFT
### LoRA






### Deepspeed


