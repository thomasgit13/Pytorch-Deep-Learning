# Transformer Architecture 
![image](.images/transfomer_architecture.png) 
We have an input sequence and a target sequence of length 8 (8 tokens). For the training, we take the batch size as 64 and model dimension as 512. So the first batch input will be of shape (64,8,512). To this input we add positional embeddings and the output will be of same dimension

Now comes the first encoder which takes the (64,8,512) input. We have 8 tokens, so we need to get a new representation for this vector by taking some weighted aggregation of other tokens. Now we make 3 copies of input(queries,keys,values).

![image](./images/self_attention.png)
- Queries are $q_1,q_2,q_3,...,q_8$ each query has  dimension $d_q$.
- Values  are $v_1,v_2,...,v_8$ each value has dimension $d_v$.
- Keys are $k_1,k_2,... ,k_8$ each key has dimension $d_k$.
- $T = 8; d_q;d_k;d_v,d_m$
- $Q=(B,T,d_q) ; K=(B,T,d_k); V = (B,T,d_v)$

We compute the attention weights as follows; 

$$
Attention(Q,K)=softmax(\frac{QK^T}{\frac{1}{\sqrt{d_k}}})
$$

Attention weight matrix will be of shape $(T,T)$. This weight will be multiplied with the value vectors. To get all the token representation. After multiplied with the value vector $V$, the output will be of shape $(B,T,d_v)$. 

This output will pass through multiple encoder layers and finally a full connected layer, and the final output will of shape $(B,T,d_m)$. 

This completes the encoder part. 

In the decoder part, we have a masked self attention layer first. Remember, in the training stage, decoder predicts the outputs all at ones using masked attention. But in inference or testing stage it predicts the outputs one by one. In masked attention, each token attention vector is created by considering all the tokens from the past. The steps in decoder part are token embedding, then positional embedding, addition of both, masked multi-headed self attention, encoder-decoder multi-headed cross attention, linear and finally the softmax layer. 

We take the target sequence first, then we apply positional embedding. 

Then it creates a masked self attention matrix. We take this output as query matrix for the next multi-headed attention layers.

Finally the ouputs will pass through linear and softmax layers.