import numpy as np
 from keras.layers import Dense,LSTM
 from keras.models import Sequential
 from sklearn.metrics import accuracy_score
 from tensorflow import set_random_seed
 from keras.models import load_model

 np.random.seed(123)
 set_random_seed(2)


 total_vocab=12101

 train_x=np.load("train_x1.npy")
 train_seq=np.load("train_y1.npy")
 test_x=np.load("test_x1.npy")
 test_seq=np.load("test_y1.npy")

 print(np.shape(train_x))
 print(np.shape(test_x))


 # One hot representation for the targets
 def one_hot(seq,total_vocab):
     seq_one_hot=np.zeros([len(seq),total_vocab])
     for i in range(0,len(seq)):
         seq_one_hot[i][seq[i]]=1
     return seq_one_hot

  # Model architecture
 def model_arch():
     #model=Sequential()
     main_input = Input(shape=(5,82), name='main_input')
     #model.add(LSTM(64,input_shape=(5,32)))
     lstm_out = LSTM(32)(main_input)
     auxiliary_input = Input(shape=(32,), name='aux_input')
     x = keras.layers.concatenate([lstm_out, auxiliary_input])

     # We stack a deep densely-connected network on top
     x = Dense(64, activation='relu')(x)
     #model.add(Dense(20,activation='relu'))
     #model.add(Dense(total_vocab,activation='softmax'))
     main_output = Dense(total_vocab, activation='softmax', name='main_output')(x)
     model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)
     model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='ADAM')
     return model

 # Training the model
 def model_fit(model,train_x,train_seq,total_vocab):
     train_y=one_hot(train_seq,total_vocab)
     print("model is building")
     model.fit(batch_size=64,epochs=10,x=[train_x,train_u],y=train_y)
     print("model building done")
     model.save('keras_model.h5')
     return model


 # Hit rate at 1 on test data
 def hit_rate_at_1(prediction,actual):
     return accuracy_score(prediction,actual)

 # Hit rata at 5 on test data
 def hit_rate_at_5(pred,actual):
     predics = []
     for i in range(0, len(pred)):
         predics.append(np.argsort(pred[i])[-5:])
     count = 0
     for i in range(0, len(predics)):
         if actual[i] in predics[i]:
             count = count + 1

     return count/len(actual)

 # Hit rate at 10 on test data
 def hit_rate_at_10(pred, actual):
     predics = []
     for i in range(0, len(pred)):
         predics.append(np.argsort(pred[i])[-10:])
     count = 0
     for i in range(0, len(predics)):
         if actual[i] in predics[i]:
             count = count + 1

     return count /len(actual)
     def model_predict(model,test_x,test_seq):
     pred=model.predict(x=test_x)
     preddy=np.argmax(a=pred,axis=1)

     print(hit_rate_at_1(preddy,test_seq))
     print(hit_rate_at_5(pred, test_seq))
     print(hit_rate_at_10(pred, test_seq))




 #model = load_model('keras_model.h5')
 model = model_arch()
 model=model_fit(model,train_x,train_seq,total_vocab)

 model_predict(model,test_x,test_seq)

 print("Done")
