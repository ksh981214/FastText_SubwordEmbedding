# FastText_SubwordEmbedding

Implement SkipGram with Negative Sampling, Subsampling and SubwordEmbedding(FastText) using character n-grams in word2vec.py

referenced by [Piotr Bojanowski∗, Edouard Grave∗, Armand Joulin, Tomas Mikolov, “Enriching Word Vectors with Subword Information”, ACL 2017](https://arxiv.org/pdf/1607.04606.pdf)

- Korea University Information Retrieval(COSE 472) Assignment5

-----

![ft](https://user-images.githubusercontent.com/38184045/71542465-45c5ee00-29aa-11ea-90e6-ef4fe4131546.PNG)

- this model use **2,3,4** and special grams

If you run "word2vec.py", you can train and test your models.

How to run

```python
python word2vec.py [mode] [partition] [update_system] [sub_sampling]
```
- mode
	- "SG" for skipgram only
 
- partition
	- "part" if you want to train on a part of corpus (fast training but worse performance) 
	- "full" if you want to train on full corpus (better performance but very slow training)
 
- update_system
 	- "NS" for Negative Sampling **only**

- sub_sampling
 	- True or False
	

***

Result

<div>
<img src="https://user-images.githubusercontent.com/38184045/71542467-4a8aa200-29aa-11ea-9487-a6a1d09da79a.PNG" width="50%">
<img src="https://user-images.githubusercontent.com/38184045/71542468-4a8aa200-29aa-11ea-90d5-1dc18cc6caa5.PNG" width="50%">
</div>

<div>
<img src="https://user-images.githubusercontent.com/38184045/71542469-4a8aa200-29aa-11ea-9b3e-2affee8ddf23.PNG" width="50%"> 
<img src="https://user-images.githubusercontent.com/38184045/71542470-4b233880-29aa-11ea-9bfb-47ff6ab68c29.PNG" width="50%">
</div>

<div>
<img src="https://user-images.githubusercontent.com/38184045/71542466-49f20b80-29aa-11ea-9a87-0710d3b0e318.PNG" width="50%">
</div>

