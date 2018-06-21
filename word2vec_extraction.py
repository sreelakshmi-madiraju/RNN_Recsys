from keras.models import load_model
import numpy as np
from gensim.models import Word2Vec
import _pickle as cPickle
from keras.models import Input,Model
from keras.layers import Dense
from collections import defaultdict
from keras import backend as K
model1 = Word2Vec.load('word2vec_model')
with open('tot_x1.pkl','rb') as f:
	new_list=cPickle.load(f)
#metadata contains one-hot vectors of features for each item
with open('metadata.pkl','rb') as f:
    meta_data=cPickle.load(f)
item_values=[]
for k,v in meta_data.items():
    if len(v)!=0:
        item_values.append(v[1:])
item_values=np.array(item_values)
print((np.shape(item_values)))

# inp=Input(shape=(2083,))
# layer1=Dense(units=500)(inp)
# layer2=Dense(units=100)(layer1)
# layer3=Dense(units=500)(layer2)
# layer4=Dense(units=2083)(layer3)
# model=Model(inputs=inp,outputs=layer4)
# model.compile(loss='mse',optimizer='ADAM')
# model.fit(batch_size=128,epochs=5,x=item_values,y=item_values)
# model.save('input_modified_meta.h5')
model = load_model('input_modified_meta.h5')
# auto encoder to get dense embeddings for meta data
get_layer2=K.function([model.layers[0].input],[model.layers[2].output])
densemeta=get_layer2([item_values])[0]
print(np.shape(densemeta))
print(densemeta[0])
w2v_data=[]
i=0
for k,v in meta_data.items():
    if len(v)!=0:
        meta_data[k]=densemeta[i]
    i=i+1
del item_values
w2v_data=[]
index=[]
for i in range(0,len(new_list)):
    seq_vec=[]
    for j in range(0,len(new_list[i])):
        q = np.concatenate([model1.wv[new_list[i][j]], meta_data[float(new_list[i][j])]])
        #print(np.shape(model1.wv[new_list[i][j]]))
        #print(np.shape(meta_data[float(new_list[i][j])]))
        if len(q)==150:
            seq_vec.append(q)
    if len(seq_vec)==9:
        w2v_data.append(seq_vec)
        index.append(i)

print(np.shape(w2v_data))
w2v_data=np.asarray(w2v_data)
print(np.shape(w2v_data))
train_x=w2v_data[0:52192]
test_x=w2v_data[52192:]
# print(np.shape(train_x))
# print(np.shape(test_x))

with open('train_x_modified.pkl','wb') as f:
    cPickle.dump(train_x,f,protocol=4)
with open('test_x_modified.pkl','wb') as f:
	cPickle.dump(test_x,f,protocol=4)
