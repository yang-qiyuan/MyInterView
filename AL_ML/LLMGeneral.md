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
It is an IO-aware exact attention algorithm that reduces memory reads/writes between GPU high-bandwidth memory (HBM) and on-chip SRAM by utilizing tiling techniques.

![flash attn](../pics/flashattn.png)
1. **Tiling**: Restructure Algorithm to load block by block from HBM to SRAM to compute attention. It decomposed softmax into small blocks and then rescale it. (One key point is the calculation of softmax without overflow).
   - Load inputs by blocks from HBM to SRAM
   - On the chip, compute attn output wrt the block
   - Update output on HBM by scaling
2. **Recomputation**: Do not store attn. matrix from forward, recompute it in backward. Efficient since normalization factors have been stored previously.  
3. 


### Multihead Latent Attention
A picture of types of attn:
![mqa](../pics/mqa.png)

MLA seeks a balance between memory efficiency and modeling accuracy.

The basic idea of MLA is to compress the attention input $h_t$ into a low-dimensional latent vector with dimension $d_c$, where $d_c$ is much lower than the original ($h_n$ · $d_h$). 

**Problems with RoPE:**
It has problems when applied RoPE to it since when multiplying Q and K together, the up-projection matrix should be contracted. However, since RoPE are index-dependent, it cannot did so. 

![drope](../pics/decoupled_rope.png)

Instead, the author decoupled the RoPE into two separate vectors and concatenate with the key and value.

![mla](../pics/mla.png)


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
   
2. **The size of the MLP** is: $d_{h}*d_{ffn} + d_{h} + d_{h}*d_{ffn} + d_{ffn}$

![layernorm](../pics/layernorm.png)
3. **The size of Layer Norm** is: $\beta, \gamma$ = $d_{h}$, which is $d_{h}*2$
4. **Output size** is: $d_{h}*d_{vocab}$


## Tokenizer
### Byte Pair Encoding
### Sentence Piece

# Model Training
## Pretraining
### Initialization
1. Random Initialization
2. Xavier and Kaiming Initialization

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

#### Differences between pre norm and post norm


#### Post Deepnorm



### Warm up

### ZeRO
#### ZeRO 1
#### ZeRO 2
#### ZeRO 3

## Finetuning
### Difference between LoRA and SFT
### LoRA


## Inference
### What is speculative decoding?

### Why LLM repeats themselves, and how to solve this problem?



### Deepspeed


