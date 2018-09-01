from gensim.models import Word2Vec
 from sklearn.preprocessing import LabelEncoder
 import numpy as np
 import _pickle as cPickle
 np.random.seed(123)

 # Model for numerical representation of the products
 def word2vec_model(new_list):
     model = Word2Vec(new_list,size =50,window = 3,min_count =1)
     return model

 # Last item purchased by the user in each sequence is taken as target
 def get_target_data(new_list):
     target = []
     for i in range(0, len(new_list)):
         q=len(new_list[i])
         for j in range(0,q):
             if j+5<q:
                 target.append(new_list[i][j+5])

     return target

 # Numbering every unique product to create one hot for target
 def num_products(model,new_list):
     entire_products=[]
     for key,value in model.wv.vocab.items():
         entire_products.append(key)

     print(len(entire_products))
     product_label=LabelEncoder()
     product_label.fit(entire_products)
     target=get_target_data(new_list)
     target_int = product_label.transform(target)
     return target_int

 #Creating five length sequences
 def five_len_sequences(new_listy):
     with open('./user_item_seq','rb') as f:
         user_item=cPickle.load(f)
     with open('./user_embed','rb') as f:
         user_embed=cPickle.load(f)

     user_input=[]
     tot_x=[]
     for i in range(0,len(new_listy)):
         q = len(new_listy[i])
         for k,v in user_item.items():
             if [new_listy[i]]==v:
                 user=k
         for j in range(0,q):
             if j+5<q:
                 tot_x.append(new_listy[i][j:j+5])
                 user_input.append(user_embed[user])
                 
     with open('./user_input.pkl','wb')as f:
         cPickle.dump(user_input,f)
     return tot_x



 new_list=np.load('user_purchase_seq.npy')
 target_data=get_target_data(new_list)
 np.save('target_data.npy',target_data)

 model=word2vec_model(new_list)
 #model.save('word2vec_model')
 y_seq=num_products(model,new_list)
 x_seq=five_len_sequences(new_list)
 print("Word2vec model built.")
 print("Labelled target sequences")


 np.save('tot_x_seq1.npy',x_seq)
 np.save('tot_y1.npy',y_seq)
