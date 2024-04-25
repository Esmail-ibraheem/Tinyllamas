## Instruct-Llama: Fine-Tuning Llama and Transformer Models with Reinforcement Learning from Human Feedback
--- 
**Contents: 
- [[#Transformers]]:  
	- [[#^bd33fb ]] 
	- [[#^20bbbe]]  
- [[#Llama2 model]]   
- [[#RLHF]]:   
	- Proximal Policy Optimization algorithm
	- Direct Preference Optimization algorithm 
	- Supervised Fine-tuning  

---
### Transformers:

**Abstract.** The Transformer neural network is a powerful deep learning model that was introduced in a landmark paper titled **"attention is all you need"** by Vaswani et al. in 2017. It revolutionized the field of natural language processing (NLP) and has since found applications in various other domains. The Transformer architecture is based on the concept of attention, enabling it to capture long-range dependencies and achieve state-of-the-art performance on a wide range of tasks.
The transformer is a neural network component that can be used to learn useful represen tations of sequences or sets of data-points [Vaswani et al., 2017]. The transformer has driven recent advances in natural language processing [Devlin et al., 2019], computer vision [Dosovitskiy et al., 2021], and spatio-temporal modelling [Bi et al., 2022].  ^bd33fb

![[Pasted image 20240418142850.png]]

**Introduction.** Before the emergence of Transformer Neural Networks (TNNs), Recurrent Neural Networks (RNNs) were commonly employed for sequential processing tasks, including machine translation. However, RNNs were characterized by slow processing speeds, limited accuracy, and challenges in handling large datasets. ^20bbbe

**here how RNN woks:** 
 is designed to process sequential data, where the current input not only depends on the current state but also on the previous inputs and states.

suppose we have this sentence "I work at the university.", and we want to translate it to Arabic "انا اعمل في الجامعة" .

In the translation task, the RNN analyzes each word ('I', 'work', 'at', 'the', 'university') one by one, updating the hidden state at each step. The output at each time step is influenced by the current word and the hidden state, which captures the historical information from previous words. The final output is a sequence of translated words ('انا', 'اعمل', 'في', 'الجامعة') in Arabic.
![[Pasted image 20240107144003.png]]
#### Problems with RNN:
1. Slow computation for long sequences 
2. Vanishing or exploding gradients 
3. Difficulty in accessing information from long time ago
4. Complexity per layer: $O(nd^2)$, mean while transformer's is $O(n^2d)$

Indeed, RNNs tend to be slow and can struggle with handling large datasets, which can lead to potential confusion or difficulties in processing extensive data. However, 

the Transformer Neural Network (TNN) introduced a breakthrough solution called "Self-Attention" in the paper "Attention is all you Need." This innovation addressed these issues and paved the way for subsequent advancements such as GPT, Bert, **Llama**, stable diffusion, and more.


### Detailed explanation:
![[Pasted image 20240424111245.png]]

so first we have the left architecture which is the "encoder" and the right is the "decoder":

1. **Input Embeddings:**  
	- Word Embedding: Represent each word as a “vector” of numbers
	
    The input sequence is transformed into fixed-dimensional embeddings, typically composed of word embeddings and positional encodings. Word embeddings capture the semantic meaning of each word. 
     ![[Pasted image 20240107144248.png]]
   
2. while **positional encodings** indicate the word's position in the sequence using the sin and cos waves.
      ![[Pasted image 20240107144329.png]]
    $$\text{PE}(i,\delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}$$
     ![[Pasted image 20240107144405.png]]
     ### Why trigonometric functions? 
     
     Trigonometric functions like cos and sin naturally represent a pattern that the model can recognize as continuous, so relative positions are easier to see for the model. By watching the plot of these functions, we can also see a regular pattern, so we can hypothesize that the model will see it too.
     
3. **Encoder and Decoder:**  
    The Transformer model consists of an encoder and a decoder. Both the encoder and decoder are composed of multiple layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network.
    
      - **_Encoder:_** The encoder takes the input sequence and processes it through multiple layers of self-attention and feed-forward networks. It captures the contextual information of each word based on the entire sequence.
        ![[Pasted image 20240424114052.png]]
      - **_Decoder:_** The decoder generates the output sequence word by word, attending to the encoded input sequence's relevant parts. It also includes an additional attention mechanism called "encoder-decoder attention" that helps the model focus on the input during decoding.
		![[Pasted image 20240424114121.png]]
1. **Self-Attention Mechanism:**   
	- __firs what is self attention:__ it is the core of the Transformer model is the self-attention mechanism. It allows each word in the input sequence to attend to all other words, capturing their relevance and influence, works by seeing how similar and important each word is to all of the words in a sentence, including itself.
				![[Pasted image 20240424104713.png]]
	- **Second the Mechanism:** 
		- **_Multi-head attention in the encoder block_:** plays a crucial role in capturing different types of information and learning diverse relationships between words. It allows the model to attend to different parts of the input sequence simultaneously and learn multiple representations of the same input.
	
		- **_Masked Multi-head attention in the decoder block_:** the same as Multi-head  attention in the encoder block but this time for the translation sentence, is used to ensure that during the decoding process, each word can only attend to the words before it. This masking prevents the model from accessing future information, which is crucial for generating the output sequence step by step.
				![[Pasted image 20240424110446.png]]
		
		- _**Multi-head attention in the decoder block_:** do the same as the Multi-head attention in the encoder block but between the input sentence and the translation sentence, is employed to capture different relationships between the input sequence and the generated output sequence. It allows the decoder to attend to different parts of the encoder's output and learn multiple representations of the context.
![[Pasted image 20240424110024.png]]
![[Pasted image 20240424110216.png]]
	
5. **Feed Forward in two blocks:** it is just feed forward neural network but in this paper the neurons are 2048. 

6. **Add & Normalization.** 
    $$Z^{(i)}_{\text{norm}} = Z^{(i)} - \frac{\mu}{\sqrt{\sigma^2 + \epsilon}}$$
    Optional using Learnable parameters: $$Z^{\tilde{(i)}} = \gamma Z_{\text{norm}}^{(i)} + \beta$$
---
### self attention mechanism:

The core of the Transformer model is the self-attention mechanism. It allows each word in the input sequence to attend to all other words, capturing their relevance and influence. Self-attention computes three vectors for each word: Query, Key, and Value.

![[Pasted image 20240424105158.png]]
- Allows to “focus attention” on particular aspects of the input text

- Done by using a set of parameters, called "weights," that determine how much attention should be paid to each input at each time step

- These weights are computed using a combination of the input and the current hidden state of the model

- Attention weights are computed (dot product of the query, key and value matrix), then a Softmax function is applied to the dot product

 - Query (Q): Each word serves as a query to compute the attention scores.
	- Q: what I am looking for.
 - Key (K): Each word acts as a key to determine its relevance to other words.
	 - K: what I can offer.
 - Value (V): Each word contributes as a value to the attention-weighted sum.
	  - what I actually offer. 

### Analogy for Q, K, V:

Library system

Imagine you're looking for information on a specific topic (query)

Each book in the library has a summary (key) that helps identify if it contains the information you're looking for

Once you find a match between your query and a summary, you access the book to get the detailed information (value) you need
   ![[Pasted image 20231224165535.png]] 
   ![[Pasted image 20231224165611.png]]



---
 ![[Pasted image 20231221101118.png]]
Attention vector for every word using this formula: 
$$Z = \text{softmax}\left(\frac{QK^T}{\sqrt{\text{Dimension of vector } Q, K \text{ or } V}}\right)V$$
Self-attention is calculated by taking the dot product of the query and key, scaled by a factor, and applying a softmax function to obtain attention weights. These attention weights determine the importance of each word's value for the current word. 
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$$\text{self attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{model}}}}+M\right)$$
### Output:

The final layer of the decoder is a linear projection followed by a softmax activation function. It produces a probability distribution over the vocabulary, allowing the model to generate the output word by sampling from this distribution.

### Softmax:
![[Pasted image 20240424113916.png]]
The softmax function is a mathematical function that converts a vector of K real numbers into a probability distribution of K possible outcomes. It is a generalization of the logistic function to multiple dimensions, and used in multinomial logistic regression. The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes. The formula for the standard (unit) softmax function is as follows:
$$\sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$
​
### Linear 
convert the embeddings to word again (**_it just has weights not biases._**)  


---
### Llama2 model:

**Introduction**. Large Language Models (LLMs) have shown great promise as highly capable AI assistants that excel in complex reasoning tasks requiring expert knowledge across a wide range of fields, including in specialized domains such as programming and creative writing. They enable interaction with humans through intuitive chat interfaces, which has led to rapid and widespread adoption among the general public. The capabilities of LLMs are remarkable considering the seemingly straightforward nature of the training methodology. Auto-regressive transformers are pretrained on an extensive corpus of self-supervised data, followed by alignment with human preferences via techniques such as Reinforcement Learning with Human Feedback (RLHF). Although the training methodology is simple, high computational requirements have limited the development of LLMs to a few players. There have been public releases of pretrained LLMs (such as BLOOM(Scao et al., 2022), LLaMa-1 (Touvron et al., 2023), and Falcon (Penedo et al., 2023)) that match the performance of closed pretrained competitors like GPT-3 (Brown et al., 2020) and Chinchilla (Hoffmannet al., 2022), but none of these models are suitable substitutes for closed “product” LLMs, such as ChatGPT, BARD, and Claude. These closed product LLMs are heavily fine-tuned to align with human preferences, which greatly enhances their usability and safety. This step can require significant costs in compute and humanannotation, and is often not transparent or easily reproducible, limiting progress within the community to advance AI alignment research.

In this work, I develop and Llama 2 model from scratch, a pretrained and fine-tuned LLM, Llama 2 at scales up to 7B parameters. 

![[Pasted image 20240424153315.png]]

Figure : Training of Llama 2-Chat: This process begins with the pretraining of Llama 2 using publicly available online sources. Following this, we create an initial version of Llama 2-Chat through the application of supervised fine-tuning. Subsequently, the model is iteratively refined using Reinforcement Learning with Human Feedback (RLHF) methodologies, specifically through rejection sampling and Proximal Policy Optimization (PPO). Throughout the RLHF stage, the accumulation of iterative reward modeling data in parallel with model enhancements is crucial to ensure the reward models remain within distribution.

---
## Llama2 architecture:
![[Pasted image 20240424154246.png]]

 1. RMS Normalization 
	 
	 Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and significantly slows the underlying network, e.g. RNNin particular, RMSNorm regularizes the summed inputs to a neuron in one layer ac cording to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm.
	 
	Awell-known explanation of the success of LayerNorm is its re-centering and re-scaling invariance property. The former enables the model to be insensitive to shift noises on both inputs and weights, and the latter keeps the output representations intact when both inputs and weights are randomly scaled.
	
	RMSNorm which only focuses on re-scaling invariance and regularizes the summed inputs simply according to the root mean square (RMS) statistic:
	$$a_i' = \frac{a_i}{\text{RMS}(a)} \cdot g_i, \quad \text{where } \text{RMS}(a) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}
$$
	 Intuitively, RMSNorm simplifies LayerNorm by totally removing the mean statistic at the cost of sacrificing the invariance that mean normalization affords.
	 
	**Why RMSNorm?**
	- Requires less computation compared to Layer Normalization. 
	- It works well in practice.
	 
 2. Rotary Positional Embeddings 
	 
	 - **Absolute Positional Encodings** 
	 
		 are fixed vectors that are added to the embedding of a token to represent its absolute position in the sentence. So, it deals with **one token at a time**. You can think of it as the pair (latitude, longitude) on a map: each point on earth will have a unique pair.
	 
	 - **Relative Position Encoding**
	 
		Relative positional encodings, on the other hand, deals with **two tokens** at a time and it is involved when we calculate the attention: since the attention mechanism captures the “intensity” of how much two words are related two each other, relative positional encodings tells the attention mechanism the distance between the two words involved in it. So, given two tokens, we create a vector that represents their distance.
		[Shaw et al. (2018)](https://arxiv.org/abs/1803.02155)) incorporated relative positional information into $𝑊^𝑘$ and $𝑊^𝑣$. Maximum relative position is clipped to a maximum absolute value of $𝑘$ and this clipping operation enables the model to generalize to unseen sequence lengths. Therefore, $2𝑘+1$ unique edge labels are considered and let us denote $\mathbf{P}^k, \mathbf{P}^v \in \mathbb{R}^{2k+1}$ as learnable relative position representations.
		$$A_{ij}^k = P^k_{\text{clip}(j - i, k)} \quad
A_{ij}^v = P^v_{\text{clip}(j - i, k)} \quad
\text{where }\text{clip}(x, k) = \text{clip}(x, -k, k)$$
		[Transformer-XL](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#transformer-xl) ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)) proposed a type of relative positional encoding based on re-parametrization of dot-product of keys and queries. To keep the positional information flow coherently across segments, Transformer-XL encodes the _relative_ position instead, as it could be sufficient enough to know the position offset for making good predictions, i.e. $i-j$, between one key vector $\mathbf{k}_{\tau, j}$ and its query $\mathbf{q}_{\tau, i}$.
		
		If omitting the scalar 1/𝑑𝑘 and the normalizing term in softmax but including positional encodings, we can write the attention score between query at position 𝑖 and key at position 𝑗 as:
		$$\begin{aligned}
a_{ij} 
&= \mathbf{q}_i {\mathbf{k}_j}^\top = (\mathbf{x}_i + \mathbf{p}_i)\mathbf{W}^q ((\mathbf{x}_j + \mathbf{p}_j)\mathbf{W}^k)^\top \\
&= \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top
\end{aligned}$$
		Transformer-XL reparameterizes the above four terms as follows:
			$$a_{ij}^\text{rel} = 
	\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{content-based addressing} + 
	\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{content-dependent positional bias} + 
	\underbrace{ \color{red}{\mathbf{u}} \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{global content bias} + 
	\underbrace{ \color{red}{\mathbf{v}} \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{global positional bias}$$
	- **Rotary Position Embedding**
		
		Rotary position embedding (_RoPE_; [Su et al. 2021](https://arxiv.org/abs/2104.09864)) encodes the absolution position with a [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) and multiplies key and value matrices of every attention layer with it to inject relative positional information at every layer.
		
		When encoding relative positional information into the inner product of the $𝑖-th$ key and the $𝑗-th$ query, we would like to formulate the function in a way that the inner product is only about the relative position 𝑖−𝑗. Rotary Position Embedding (RoPE) makes use of the rotation operation in Euclidean space and frames the relative position embedding as simply rotating feature matrix by an angle proportional to its position index.
		
		Given a vector 𝑧, if we want to rotate it counterclockwise by 𝜃, we can multiply it by a rotation matrix to get 𝑅𝑧 where the rotation matrix 𝑅 is defined as:
		$$R = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}$$
					![[Pasted image 20240424165309.png]]
		When generalizing to higher dimensional space, RoPE divide the 𝑑-dimensional space into 𝑑/2 subspaces and constructs a rotation matrix 𝑅 of size 𝑑×𝑑 for token at position 𝑖:
		$$R^d_{\Theta, i} = \begin{bmatrix}
\cos i\theta_1 & -\sin i\theta_1 & 0 & 0 & \dots & 0 & 0 \\
\sin i\theta_1 & \cos i\theta_1 & 0 & 0 & \dots & 0 & 0 \\
0 & 0 & \cos i\theta_2 & -\sin i\theta_2 & \dots & 0 & 0 \\
0 & 0 & \sin i\theta_2 & \cos i\theta_2 & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots & \cos i\theta_{d/2} & -\sin i\theta_{d/2} \\
0 & 0 & 0 & 0 & \dots & \sin i\theta_{d/2} & \cos i\theta_{d/2} \\
\end{bmatrix}$$
		where in the paper we have Θ=𝜃𝑖=10000−2(𝑖−1)/𝑑,𝑖∈[1,2,…,𝑑/2]. Note that this is essentially equivalent to sinusoidal positional encoding but formulated as a rotation matrix.
		
		Then both key and query matrices incorporates the positional information by multiplying with this rotation matrix:
		$$\begin{aligned}
& \mathbf{q}_i^\top \mathbf{k}_j = (R^d_{\Theta, i} \mathbf{W}^q\mathbf{x}_i)^\top (R^d_{\Theta, j} \mathbf{W}^k\mathbf{x}_j) = \mathbf{x}_i^\top\mathbf{W}^q R^d_{\Theta, j-i}\mathbf{W}^k\mathbf{x}_j \\
& \text{ where } R^d_{\Theta, j-i} = (R^d_{\Theta, i})^\top R^d_{\Theta, j}
\end{aligned}$$
		![[Pasted image 20240424161444.png]]

3. KV-Cache 
	
	Recall the definition of Attention given in the [“Attention Is All You Need”](https://arxiv.org/pdf/1706.03762.pdf) paper:

	$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

	where 𝑄, 𝐾, and 𝑉 are three matrices that are trained during the training process. The embeddings of each token (a vector) is multiplied by these three matrices to obtain three vectors 𝑞𝑛, 𝑘𝑛, and 𝑣𝑛.
	
	When computing self-attention, we compute the dot product of the query vector 𝑞𝑛 with the key vector of every other token before it in the input sequence $𝑘_𝑛,𝑘_{𝑛+1},…,𝑘_𝑁$.
	
	Each product 𝑞𝑖𝑇⋅𝑘𝑗 is divided by the square root of the dimension of the key vectors 𝑑𝑘 in order to have more stable gradients. Eventually, everything is passed through a softmax to normalize the scores:
	$$a_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{t=1}^{i}\exp(q_i^T k_t / \sqrt{d_k})}$$
	
	The final output is derived by computing the weighted average over the value vectors:
	$$o_i = \sum_{j=1}^{i} a_{ij} v_j$$
	**The autoregressive nature of transformers**
	
	transformer-based models are **autoregressive models**, meaning essentially that they use the past to predict the future.
	
	Given a prompt $(x_1, …, x_n)$
	
	Since the tokens (𝑥1,…,𝑥𝑛) are all known, computing 𝑃(𝑥𝑛+1|𝑥1,…,𝑥𝑛) can be made with matrix-matrix multiplication and thus benefit from GPU parallelism.

	Instead, when we get to compute the remaining tokens 𝑃(𝑥𝑛+𝑡+1|𝑥1,…,𝑥𝑛+𝑡), the data dependency forces us to use a matrix-vector multiplication, which is less efficient and leads to an **underutilization of the GPU**.
	
	I**n the process we described above**, one can notice that the key and value vectors 𝑘1,…,𝑘𝑛+𝑡−1 and 𝑣1,…,𝑣𝑛+𝑡−1 seem to be re-computed every time a new token is taken into consideration. Of course, this would be a waste of resources.

	Consider the below illustration:
	![[Pasted image 20240425100728.png]]

	The 𝐾 and 𝑉 matrices contain information about all the sequence, while the query vector contains just the information about the last token. The dot product between 𝑞 and 𝐾 corresponds to doing attention between the last token (i.e. “blue” in our example) and all the previous ones.

	Note two things:
	- during the sequence generation one token at a time, the two matrices 𝐾 and 𝑉 do not change very much
	- once we computed the embedding for the new token, it’s not going to change, no matter how many more tokens we generate

	That is why the key and value vectors of existing tokens are often cached for generating future tokens. This approach leads to what is called the **KV cache**. Note that the KV cache of one token depends on all its previous tokens, hence if we have the same token appearing in two different positions inside the sequence, the corresponding KV caches will be different as well.
	![[Pasted image 20240425100844.png]]
	**How much memory does KV cache use?**

	Let’s consider a 13B parameter [OPT model](https://arxiv.org/pdf/2205.01068.pdf)
	$memory\_usage\_per\_token = num\_vectors * hidden\_state\_size * num\_layers * precision\_(bytes) = 2 * 5120 * 40 * 2 = 800KB$
	
	where num_vectors refers to the key and value vectors.
	
	In OPT a sequence can be made of up to 2048 tokens, hence we would need 800∗2048≈1.6GB per single request.
	
4. Grouped-Query Attention
	
	The standard practice for autoregressive decoding is to cache the keys and values of the previous tokens in the sequence to speed up attention computation. However, as the context window or batch size increases, the memory cost associated with the size of the key-value cache(kv cache) in the multi-head attention(MHA) model significantly increases.

	Multi-Query attention(MQA) is a mechanism that uses only a single key-value head for multiple queries, which can save memory and greatly speed up decoder inference.
	
	However, MQA may lead to a decrease in quality. In fact, we not only want fast inference, but also want the quality to be on par with MHA, so Grouped-query attention(GQA)[1] comes into play.

	Grouped-query attention(GQA) is an interpolation of multi-query and multi-head attention. It achieves a quality similar to multi-head attention while maintaining a comparable speed to multi-query attention.
	
	Grouped-query attention divides query heads into G-groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. following Figure shows a comparison of grouped-query attention and multi head/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by mean pooling all the original heads within that group. An intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, rep resents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases.
	![[Pasted image 20240425092412.png]]
	Overview of grouped-query method. Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.
	
	**MHA vs GQA vs MQA :**
		![[Pasted image 20240425093345.png]]
	**Uptraining steps** Figure shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA. First, we note that GQAalready achieves reasonable performance af ter conversion while MQA requires uptraining to be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%.
		![[Pasted image 20240425093517.png]]
	Time per sample for GQA-XXL as a function of the number of GQA groups with input length 2048 and output length 512. Going from 1 (MQA) to 8 groups adds modest inference overhead, with increasing cost to adding more groups.
	
	**Number of groups**: Figure demonstrates the effect of the number of GQA groups on inference speed. For larger models the memory band width overhead from the KV cache is less con straining (Shazeer, 2019), while the reduction in key-value size is sharper due to the increased number of heads. As a result, increasing the number of groups from MQA only results in modest slow downs initially, with increasing cost as we move closer to MHA. We selected 8 groups as a favor able 
	middle ground.

|         MHA          |                  GQA                  |         MQA          |
|:--------------------:|:-------------------------------------:|:--------------------:|
|     High quality     | A good compromise between quality and |   Loss in quality    |
| Computationally slow |                 speed                 | Computationally fast |
	
>[!Note] MHA enables a nuanced understanding of the relationships between different parts of the input. Nevertheless, this complexity comes at a cost — a significant demand on memory bandwidth, especially during decoder inference. In multi-query attention, we average the heads for keys and values so that all query heads share the same key and value head. This is achieved by replicating the mean-pooled “head” H times, where H is the number of query heads. However, MQA is not without its drawbacks. The reduced complexity can lead to quality degradation and training instability. Grouped-query attention (GQA) is a simple approach that blends elements of multi-head attention (MHA) and multi-query attention (MQA) to create a more efficient attention mechanism.


5. SwiGLU Activation Function:
	**SwiGLU** is an activation function which is a variant of [GLU](https://paperswithcode.com/method/glu). The definition is as follows:
	$$ \text{SwiGLU}\left(x, W, V, b, c, \beta\right) = \text{Swish}_{\beta}\left(xW + b\right) \otimes \left(xV + c\right) $$
			![[Pasted image 20240425141023.png]]
	[SwiGLU](https://paperswithcode.com/method/swish) is a combination of Swish and GLU activation functions. SwiGLU is defined as follows:
	
	$$SwiGLU(x) = x * sigmoid(beta * x) + (1 - sigmoid(beta * x)) * (Wx + b)$$
	
	where 'W', 'b', and 'beta' are trainable parameters.
	
	In SwiGLU, the Swish function is used to gate the linear function of GLU. This allows SwiGLU to capture the advantages of both Swish and GLU, while overcoming their respective disadvantages.

![[Pasted image 20240425140310.png]]

---
## Llama 1 vs Llama 2
![[Pasted image 20240424154525.png]]

**Tokenizer.** We use the same tokenizer as Llama1; it employs a byte-pair encoding(BPE) algorithm (Sennrich etal.,2016) using the implementation from Sentence-Piece(KudoandRichardson,2018). AswithLlama1, we split all numbers into individual digits and use bytes to decompose unknown UTF-8characters.The total vocabulary size is 32k tokens.

---


### RLHF:
- ##### Proximal Policy Optimization algorithm
- ##### Direct Preference Optimization algorithm
- ##### Supervised Fine-tuning 

![[Pasted image 20240424183323.png]]

