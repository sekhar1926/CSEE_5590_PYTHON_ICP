import numpy as np
import pandas as pd

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM

from keras.layers.convolutional import Conv1D,MaxPooling1D,Conv2D
from keras.layers import SpatialDropout1D


np.random.seed(0)

if __name__ == "__main__":

    # load data
    train_df = pd.read_csv('train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('test.tsv', sep='\t', header=0)

    raw_docs_train = train_df['Phrase'].values
    raw_docs_test = test_df['Phrase'].values
    sentiment_train = train_df['Sentiment'].values
    num_labels = len(np.unique(sentiment_train))

    # text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')

    print("pre-processing train docs...")
    processed_docs_train = []
    for doc in raw_docs_train:
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_train.append(stemmed)

    print("pre-processing test docs...")
    processed_docs_test = []
    for doc in raw_docs_test:
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_test.append(stemmed)

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

    dictionary = corpora.Dictionary(processed_docs_all)
    dictionary_size = len(dictionary.keys())
    print("dictionary size: ", dictionary_size)
    # dictionary.save('dictionary.dict')
    # corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print("converting to token ids...")
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))

    seq_len = np.round((np.mean(word_id_len) + 2 * np.std(word_id_len))).astype(int)

    # pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)
    print(word_id_train)
    print(y_train_enc)
    # LSTM
    print("fitting LSTM ...")
    model = Sequential()

    model.add(Embedding(dictionary_size, 128, dropout=0.2))
    model.add(LSTM(128, dropout=0.4,recurrent_dropout=0.2,return_sequences=True))
    model.add(LSTM(32,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))


    hist = model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    hist = model.fit(word_id_train, y_train_enc, nb_epoch=50, batch_size=256, verbose=1)

    test_pred = model.predict_classes(word_id_test)

    print(hist.history.get('acc')[-1])
    '''#make a submission
    test_df['Sentiment'] = test_pred.reshape(-1,1) 
    header = ['PhraseId', 'Sentiment']
    test_df.to_csv('./lstm_sentiment.csv', columns=header, index=False, header=True)'''
    # save to disk
    model1_json = model.to_json()
    with open('model1.json', 'w') as json_file:
        json_file.write(model1_json)
    model.save_weights('model1.h5')
