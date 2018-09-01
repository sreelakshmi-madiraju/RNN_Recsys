import numpy as np
 from gensim.models import Word2Vec
 import _pickle as cPickle

 np.random.seed(123)
#item_emb contains neural embeddings of all items
 with open('./item_embed', 'rb') as f:
     item_list=cPickle.load(f)

 new_list=np.load('tot_x_seq1.npy')
 model = Word2Vec.load('word2vec_model')
 y_tot=np.load('tot_y1.npy')
 
 # Appending product vectors with their neural embeddings
 def w2v_data_ext(new_list):
     w2v_data=[]
     for i in range(0,len(new_list)):
         seq_vec=[]
         for j in range(0,len(new_list[i])):
             q = np.concatenate([model.wv[new_list[i][j]], item_list[new_list[i][j]]])
             if len(q)==82:
                 seq_vec.append(q)
         if len(seq_vec)==5:
             w2v_data.append(seq_vec)
     return np.asarray(w2v_data)


 # Train and test split
 def train_test_split(w2v_data,y_tot):
     train_x=w2v_data[0:69349]
     test_x=w2v_data[69349:]
     train_y=y_tot[0:69349]
     test_y=y_tot[69349:]
     print(np.shape(train_x))
     print(np.shape(train_y))
     print(np.shape(test_x))
     print(np.shape(test_y))
     return train_x,train_y,test_x,test_y


 w2vdata=w2v_data_ext(new_list)
 print(np.shape(w2vdata))
 train_x,train_y,test_x,test_y=train_test_split(w2vdata,y_tot)
 print("Train and test data saved")

 np.save("train_x1.npy", train_x)
 np.save("train_y1.npy", train_y)
 np.save("test_x1.npy", test_x)
 np.save("test_y1.npy", test_y)
