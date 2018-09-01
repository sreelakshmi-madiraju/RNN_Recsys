import pandas as pd
 import numpy as np
 import warnings
 from keras.models import load_model
 from gensim.models import Word2Vec

 from sklearn.model_selection import train_test_split
 from sklearn.metrics import precision_score
 from sklearn.metrics import accuracy_score

 np.random.seed(123)
 import keras.models as kmodels
 import keras.layers as klayers
 import keras.backend as K
 import keras
 from keras.optimizers import Adam
 from sklearn import dummy, metrics, cross_validation, ensemble
 warnings.filterwarnings('ignore')
 from sklearn.model_selection import train_test_split
 from sklearn.model_selection import TimeSeriesSplit

 df=pd.read_csv('./beauty_df.csv',header='infer')
 df=df[['reviewerID', 'asin','overall','reviewTime']]
 df.columns=['userid', 'item_id', 'rating', 'reviewTime']
 df=df.sort_values(['reviewTime'],ascending=[True])
 df.userid = df.userid.astype('category').cat.codes.values
 df.item_id = df.item_id.astype('category').cat.codes.values

 n_users, n_items = len(df.userid.unique()), len(df.item_id.unique())

 y = np.zeros((df.shape[0], 5))
 a=np.arange(df.shape[0])
 b=df.rating.values
 b=[int(i) for i in b]
 b=[i-1 for i in b]
 y[a,b]=1
 print(np.shape(y))
 
 item_input = keras.layers.Input(shape=[1])
 item_vec = keras.layers.Flatten()(keras.layers.Embedding(n_items + 1, 32)(item_input))
 item_vec = keras.layers.Dropout(0.5)(item_vec)
 
 user_input = keras.layers.Input(shape=[1])
 user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32)(user_input))
 user_vec = keras.layers.Dropout(0.5)(user_vec)
 
 
 input_vecs = keras.layers.merge([item_vec, user_vec], mode='concat')
 print(np.shape(input_vecs))

 nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
 nn = keras.layers.normalization.BatchNormalization()(nn)
 nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
 nn = keras.layers.normalization.BatchNormalization()(nn)
 nn = keras.layers.Dense(128, activation='relu')(nn)
 result = keras.layers.Dense(5, activation='softmax')(nn)
 model = kmodels.Model([movie_input, user_input], result)
 model.compile('adam', 'categorical_crossentropy')
 
 a_item_id, b_item_id, a_userid, b_userid, a_y, b_y = train_test_split(df.item_id, df.userid, y,shuffle=False)
 history = model.fit([a_item_id, a_userid], a_y,nb_epoch=5,validation_data=([b_item_id, b_userid], b_y))
 print("test error is")
 print(metrics.mean_absolute_error(np.argmax(b_y, 1)+1,np.argmax(model.predict([b_item_id, b_userid]), 1)+1))
 print("accuracy is")
 print(accuracy_score(np.argmax(b_y, 1)+1,np.argmax(model.predict([b_item_id, b_userid]), 1)+1))
 
 #getting user,item embeddings
 weights=model.get_weights()
 user_embeddings = weights[1]
 item_embeddings = weights[0]
 with open('item_embed','wb')as f:
    cPickle.dump(item_embeddings,f)
 with open('user_embed','wb')as f:
    cPickle.dump(user_embeddings,f)
    
