#importing and loading all the neccesseties
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import gradient_descent_v2 as SGD

words=[]
classes=[]
documents=[]
ignore_words=['?','!']
data_file= open('intents.json').read()
intents = json.loads(data_file)

#Preprocessing the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w= nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w,intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes= sorted(list(set(classes)))

print(len(documents),"documents=")
print(len(classes),"classes=",classes)
print(len(words),"Unique words",words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#Creating training and testing data 
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern_words= doc[0]
    pattern_words = [lemmatizer.lemmatize(words.lower()) for words in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        output_row= list(output_empty)
        output_row[classes.index(doc[1])] =1

        training.append([bag,output_row])

random.shuffle(training)
training = np.array(training,dtype=object)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
#Creating Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile Model
sgd= tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, nesterov=True, name="SGD"
)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fitting and saving Model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=40, batch_size=5, verbose=1)
model.save('Chatbot_model.h5',hist)
print("Model Created")



