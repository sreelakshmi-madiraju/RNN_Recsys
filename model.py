import numpy as np
from keras.layers import Dense,LSTM
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras import backend as K
import _pickle as cPickle
y_tot=np.load('tot_y.npy')
with open('items_index_present.pkl','rb') as f:
	index=cPickle.load(f)
y_tot=y_tot[index,:]
print(np.shape(y_tot))
train_y=y_tot[0:52192]
test_y=y_tot[52192:]
del y_tot
total_vocab=78089
with open('train_x_modified.pkl','rb') as f:
	train_x=cPickle.load(f)
train_x=np.asarray(train_x)
with open('test_x_modified.pkl','rb') as f:
	test_x=cPickle.load(f)
test_x=np.asarray(test_x)
#with open('tot_y_target.pkl','rb') as f:
#	target_int=cPickle.load(f)
model=Sequential()
model.add(LSTM(64,input_shape=(9,150)))
model.add(Dense(20,activation='relu'))
model.add(Dense(total_vocab,activation='softmax'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='ADAM')
#model.compile(loss='mse',optimizer='ADAM')

model.fit(batch_size=64,epochs=5,x=train_x,y=train_y)
model.save('rnn_model_modified2.h5')
#model = load_model('rnn_model.h5')

print("model building done")
# del train_y
# del train_x
pred=model.predict(x=test_x)

'''predic=model.predict(x=train_x)
from sklearn.neighbors import NearestNeighbors
neight=NearestNeighbors(n_neighbors=10)
total_word2vec=np.concatenate((train_y,test_y))
print(np.shape(total_word2vec))
neight.fit(total_word2vec)
neigh_5=neight.kneighbors(predic)[1]
count=0
for i in range(0,len(neigh_5)):
	if target_int[i] in neigh_5[i]:
		count=count+1
print(count)
print('accuracy',float(count/len(train_x)))

'''
preddy=np.argmax(a=pred,axis=1)
truey=np.argmax(a=test_y,axis=1)

hit_rate_at_1=accuracy_score(y_pred=preddy,y_true=truey)
print("hit_rate_@1 ",hit_rate_at_1)

predics = []
for i in range(0, len(pred)):
    predics.append(np.argsort(pred[i])[-5:])
count=0
for i in range(0,len(predics)):
    if truey[i] in predics[i]:
        count=count+1

hit_rate_at_5=count/len(test_y)
print("hit_rate_@5 ",hit_rate_at_5)

predics1 = []
for i in range(0, len(pred)):
    predics1.append(np.argsort(pred[i])[-10:])


count1=0
for i in range(0,len(predics1)):
    if truey[i] in predics1[i]:
         count1=count1+1

hit_rate_at_10=count1/len(test_y)
print("hit_rate_@10 ",hit_rate_at_10)
