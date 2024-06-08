# Xllama🦙: Extensible Language Model inspired by the original Llama model.
 
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/FeedbackTransformer/blob/main/llama2.jpg" alt="Your Image Description" width="250" height=250">
</p>


X-Llama is an advanced language model framework, inspired by the original Llama model but enhanced with additional features such as Grouped Query Attention (GQA), Multi-Head Attention (MHA), and more. This project aims to provide a flexible and extensible platform for experimenting with various attention mechanisms and building state-of-the-art natural language processing models.

**_project structure:_**
The [model](https://github.com/Esmail-ibraheem/X-Llama/blob/main/model.py) was constructed in approximately ~500 lines of code, and you have the model's [configuration](https://github.com/Esmail-ibraheem/X-Llama/blob/main/config.py).
```
X-Llama/
│
├── images/
│
├── models/
│   ├── attentions/
│   ├── rotary_embeddings/
│   └── transformer/
│
├── model
│
└── config
│
└── inference

```

---

## Features:
- **`Rotary Embeddings`:**
   - Rotary Embeddings.
   - Linear Scaling Rotary Embeddings.
   - Dynamic NTK Scaling Rotary Embeddings.
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/RoPE.png" alt="Your Image Description">
</p>

```python
LLAMA_ROTARY_EMBEDDINGS_CLASSES = {
    "rotary": LlamaRotaryEmbeddings,
    "linear": LlamaLinearScalingRotaryEmbeddings,
    "dynamic": LlamaDynamicNTKScalingRotaryEmbeddings,
	}
```

- **`LlamaChat`.**
  	![__-Llama-2-Chatbot-by-Esmail-Gumaan-and-2-more-pages-Personal-Microsoft_-Edge-2024-05-15-17-19-03](https://github.com/Esmail-ibraheem/X-Llama/assets/113830751/b52b5b68-3f5e-4cfb-9719-b0fae5fa4678)

- **`Attentions:`**
  The standard practice for autoregressive decoding is to cache the keys and values of the previous tokens in the sequence to speed up attention computation. However, as the context window or batch size increases, the memory cost associated with the size of the key-value cache(kv cache) in the multi-head attention(MHA) model significantly increases.
   - **`Multi-Head Attention(MHA)`:**\
       [Self-attention](https://github.com/Esmail-ibraheem/X-Llama/blob/main/models/transformer.py) is calculated by taking the dot product of the query and key, scaled by a factor, and applying a softmax function to obtain attention weights. These [attention](https://github.com/Esmail-ibraheem/X-Llama/blob/main/models/attentions.py) weights determine the importance of each word's value for the current word.
     $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
       <p align="center">
       <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/MHA.png" alt="Your Image Description">
     </p>

     ---
     
   - **`Grouped Query Attention(GQA), and Multi-Query Attention(MQA)`:**\
      Grouped-query attention divides query heads into G-groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. following Figure shows a comparison of grouped-query attention and multi head/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by mean pooling all the original heads within that group. An intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, rep resents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases.
      - MQA: Multi-Query attention(MQA) is a mechanism that uses only a single key-value head for multiple queries, which can save memory and greatly speed up decoder inference.
      - Fixed GQA: However, MQA may lead to a decrease in quality. In fact, we not only want fast inference but also want the quality to be on par with MHA, so Grouped-query attention(GQA) comes into play. Grouped-query attention(GQA) is an interpolation of multi-query and multi-head attention. It achieves a quality similar to multi-head attention while maintaining a comparable speed to multi-query attention.
      - `Scalable GQA:` the same as the fixed GQA but with multiple rotary embeddings.
           <p align="center">
         <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/GQA.png" alt="Your Image Description">
       </p>

     **_MHA vs GQA vs MQA:_**
	     
	|         MHA          |                  GQA                  |         MQA          |
	|:--------------------:|:-------------------------------------:|:--------------------:|
	|     High quality     | A good compromise between quality and |   Loss in quality    |
	| Computationally slow |                 speed                 | Computationally fast |
		

	<p align="center">
	  <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/MHA%2CGQA%2CMQA2.png" alt="Your Image Description" width="600" height=400">
	</p>
		 Time per sample for GQA-XXL as a function of the number of GQA groups with input length 2048 and output length 512. Going from 1 (MQA) to 8 groups adds modest inference overhead, with increasing cost to adding more groups.
	   demonstrates the effect of the number of GQA groups on inference speed. For larger models the memory band width overhead from the KV cache is less con straining (Shazeer, 2019), while the reduction in key-value size is sharper due to the increased number of heads. As a result, increasing the number of groups from MQA only results in modest slow downs initially, with increasing cost as we move closer to MHA. We selected 8 groups as a favor able 
		middle ground.

 	<p align="center">
	  <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/MHA%2CGQA%2CMQA.png" alt="Your Image Description" width="600" height=400">
	</p>
		 	shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA. First, we note that GQA already achieves reasonable performance after conversion while MQA requires uptraining to be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%.
	
    > MHA enables a nuanced understanding of the relationships between different parts of the input. Nevertheless, this complexity comes at a cost — a significant demand on memory bandwidth, especially during decoder inference. In multi-query attention, we average the heads for keys and values so that all query heads share the same key and value head. This is achieved by replicating the mean-pooled “head” H times, where H is the number of query heads. However, MQA is not without its drawbacks. The reduced complexity can lead to quality degradation and training instability. Grouped-query attention (GQA) is a simple approach that blends elements of multi-head attention (MHA) and multi-query attention (MQA) to create a more efficient attention mechanism.

```python
LLAMA_ATTENTIONS_CLASSES = {
    "GQA": LlamaScalableGroupedQueryAttention,
    "MHA": MultiHeadAttention,
    "MQA": MultiQueryAttention,
}
```

---

## installs:
install the requirements libraries:
```
pip install requirements
```
or 
```
pip install pytorch transformers
```
clone the repo
```
git clone https://github.com/Esmail-ibraheem/X-Llama.git
```
run the download shell file to download the llama 2 weights 
```
.\download.sh
```
after downloading the weights, run the inference code:
```
python inference.py
```

now you should be able to test the model, by changing the prompts to whatever you want, here I wrote some physics prompts:
```python
prompts = [
        "Simulate the motion of a projectile launched at a certain angle and velocity, including factors like gravity and air resistance.",
        "Create a program that calculates the gravitational force between two objects based on their masses and distances."
        "Develop a program to simulate the behavior of ideal gases using the laws of thermodynamics."
    ]
```

---

## Citation:
```BibTex
@misc{Gumaan2024-X-Llama,
  title   = "X-Llama",
  author  = "Gumaan, Esmail",
  howpublished = {\url{https://github.com/Esmail-ibraheem/X-Llama}},
  year    = "2024",
  month   = "May",
  note    = "[Online; accessed 2024-05-15]",
}



```
---
## Notes and Acknowledgments:
I developed this project to enhance my skills in large language models and transformers. I built the Llama model from scratch and implemented various features, including multiple attentions. Feel free to suggest any additional features you'd like, such as flash attention or related concepts. This project integrates multiple research papers.

**papers**:
- [llama 2 research paper](https://arxiv.org/abs/2307.09288)
- [attention is all you need research paper](https://arxiv.org/abs/1706.03762)
- [Grouped Query Attention research paper](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding research paper](https://arxiv.org/abs/2104.09864)

**other**
- [llama from scratch](https://youtu.be/oM4VmoabDAI?si=rDegyrnSghByUEnK)
- [huggingFace transformers lib](https://github.com/huggingface/transformers)

