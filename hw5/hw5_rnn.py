import numpy as np
import string
import sys
import keras.backend as K
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU,Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import Normalizer,normalize
from sklearn.decomposition import PCA, TruncatedSVD

K.clear_session()
# train_path = sys.argv[1]
test_path = sys.argv[1]
output_path = sys.argv[2]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 300
batch_size = 512
range_value = 0.5

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

def translate_Categorial2label(output, label):
    Y = []
    for row in output:
        find = [pos for pos,x in enumerate(row) if x==1]
        temp = []
        for i in find:
            temp.append(label[i])
        temp = [" ".join(temp)]
        Y.append(temp)
    return Y

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.3
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

#########################
###   Main function   ###
#########################
def main():
    """
    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus_temp = X_data + X_test
    # print ('Find %d articles.' %(len(all_corpus_temp)))

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=2, max_df  = 0.7)
    all_corpus_temp = vectorizer.fit_transform(all_corpus_temp)

    #####normalize
    normalizer = Normalizer(norm='l2',copy=True)
    all_corpus_temp = normalize(all_corpus_temp, norm='l2')
    all_corpus_temp = all_corpus_temp.toarray()
    X_data = all_corpus_temp[0:4964,:]
    X_test = all_corpus_temp[4964:6198,:]
    print(np.shape(X_test))
    flag = 0
    stepsize=int(1234/8)
    for i in range(8):
        if (i<7):
            print('Saving X_test array... # of file:',(i+1))
            np.savetxt(('rnntest%s.out' % (i+1)),X_test[flag:flag+stepsize,:])
        else:
            print('Saving X_test array... # of file: 8')
            np.savetxt(('rnntest%s.out' % (i+1)),X_test[flag:1234,:])
        flag += stepsize
    """
    
    X_test = np.zeros((1234,22266))
    flag = 0
    stepsize=int(1234/8)
    for i in range(8):
        if (i<7):
            print('Loading X_test array... # of file:',(i+1))
            X_test[flag:flag+stepsize,:] = np.loadtxt(('model/rnntest%s.out' % (i+1)))
        else:
            print('Loading X_test array... # of file: 8')
            X_test[flag:1234,:] = np.loadtxt(('model/rnntest%s.out' % (i+1)))
        flag += stepsize
    
    tag_list = ['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE", 'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL', 'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL', 'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR', 'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER', 'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION', 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']

    ###
    # train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    # (X_train,Y_train),(X_val,Y_val) = split_data(X_data,train_tag,split_ratio)

    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Dense(128,input_shape=(22266,),activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    """
    rms = RMSprop(lr = 0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=[f1_score])
   
    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='rnnmodel.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
    hist = model.fit(X_train, Y_train, 
                     validation_data=(X_val, Y_val),
                     epochs=nb_epoch, 
                     batch_size=batch_size,
                     callbacks=[earlystopping,checkpoint])
    """

    print ('Loading model.')
    model.load_weights('model/rnnmodel.hdf5')
    rms = RMSprop(lr = 0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=[f1_score])

    Y_pred = model.predict(X_test)

    linfnorm = np.linalg.norm(Y_pred, axis=1, ord=np.inf)
    preds = Y_pred.astype(np.float) / linfnorm[:, None]

    preds[preds >= range_value] = 1
    preds[preds < range_value] = 0

    original_y = translate_Categorial2label(preds, tag_list)

    output_file = []

    output_file.append('"id","tags"')
    for i in range (len(original_y)):
        temp = '"'+str(i)+'"' + ',' + '"' + str(" ".join(original_y[i])) + '"'
        output_file.append(temp)

        with open(output_path,'w') as f:
            for data in output_file:
                f.write('{}\n'.format(data))
    K.clear_session()

if __name__=='__main__':
    main()
