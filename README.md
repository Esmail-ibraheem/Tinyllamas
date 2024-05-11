# X-Llama🦙: Extensible Language Model inspired by the original Llama model.
 
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/FeedbackTransformer/blob/main/llama2.jpg" alt="Your Image Description" width="400" height=400">
</p>


X-Llama is an advanced language model framework, inspired by the original Llama model but enhanced with additional features such as Grouped Query Attention (GQA), Multi-Head Attention (MHA), and more. This project aims to provide a flexible and extensible platform for experimenting with various attention mechanisms and building state-of-the-art natural language processing models.

## Features:
- **`Rotary Embeddings`:**
   - Rotary Embeddings.
   - Linear Scaling Rotary Embeddings.
   - Dynamic NTK Scaling Rotary Embeddings.
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/RoPE.png" alt="Your Image Description">
</p>

- **`LlamaChat`.**
- **`Attentions:`**
  The standard practice for autoregressive decoding is to cache the keys and values of the previous tokens in the sequence to speed up attention computation. However, as the context window or batch size increases, the memory cost associated with the size of the key-value cache(kv cache) in the multi-head attention(MHA) model significantly increases.
   - **`Multi-Head Attention(MHA)`:**\
       Self-attention is calculated by taking the dot product of the query and key, scaled by a factor, and applying a softmax function to obtain attention weights. These attention weights determine the importance of each word's value for the current word.
     $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
       <p align="center">
       <img src="https://github.com/Esmail-ibraheem/X-Llama/blob/main/images/MHA.png" alt="Your Image Description">
     </p>
     
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
		





   - **`Flash-Attention`:**

---

## BibTex
```BibTex
@article{Gumaan2024-X-Llama,
  title   = "X-Llama",
  author  = "Gumaan, Esmail",
  code = "https://github.com/Esmail-ibraheem/X-Llama/tree/main",
  year    = "2024",
  month   = "Agu",
  url     = ""
}

```
---
## Notes:
