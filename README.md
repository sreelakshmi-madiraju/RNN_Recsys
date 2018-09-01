# RNN_Recsys
We find the transaction sequence of each user sorted by transaction time (w2v_before.py).  
We consider the last item in the sequence as target.  We find the vector representations of items(prod2vec embeddings) using these sequences. (word2vec.py)
we find the item and user embeddings from the transactional data using a neural network.(user_item_embeddings.py)
We represent each item with the concatenated prod2vec embeddings and neural embeddings. (word2vec_extraction.py) 
Prediction is done using LSTM model. user embeddings are added as auxillary input to the model. (model.py)
