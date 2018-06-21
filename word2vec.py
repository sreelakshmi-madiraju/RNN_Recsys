from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import numpy as np
import _pickle as cPickle
import pickle
import hickle as hkl

#new_list=np.load('user_purchase_seq.npy')
#items_pruned contains all user puchase sequences ordered by time.
#each item is purchased by min 10 users and each user purchased min 10 items
with open('./items_pruned1.pkl','rb') as f:
	new_list=pickle.load(f)
#model = Word2Vec(new_list,size =50,window = 3,min_count =1)

#model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')

entire_products=[]
for key,value in model.wv.vocab.items():
		entire_products.append(key)
#print((entire_products))

product_label=LabelEncoder()
product_label.fit(entire_products)
num_label=product_label.transform(entire_products)

target=[]
new_list1=[]
for i in range(0,len(new_list)):
	new_list1.append(new_list[i])
	target.append(new_list[i][-1])
print(len(target))
print(len(new_list1))

new_list=new_list1
target_int=product_label.transform(target)
print(len(target_int))

n_values = np.max(target_int) + 1
print(n_values,target_int)
y_tot=np.zeros((len(new_list),len(entire_products)))
print(np.shape(y_tot))
for i in range(0,len(new_list)):
	y_tot[i][target_int[i]]=1

#y_tot=np.eye(n_values)[target_int]


for i in range(0,len(new_list)):
	q=len(new_list[i])
	new_list[i]=new_list[i][q-10:q-1]

print("Word2vec model built.")
print("Labelled sequences and One hot for target data created")
with open('tot_x1.pkl','wb') as f:
   cPickle.dump(new_list,f)
#with open('tot_y1.pkl','wb') as f:
 #  cPickle.dump(y_tot,f,protocol=2)
np.save('tot_y.npy',y_tot)
