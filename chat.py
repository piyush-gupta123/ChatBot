import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import pickle

from keras.models import load_model
model = load_model('Chatbot_model.h5')
import json 
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words= clean_up_sentence(sentence)
    bag = [0]*len(words)

    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
                if show_details:
                    print("Found in bag %s" %w)

    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    Err_threshold= 0.25

    results = [[i,r] for i,r in enumerate(res) if r>Err_threshold]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# import tkinter 
# from tkinter import *

# def send():
#     msg = EntryBox.get("1.0","end-1c").strip()
#     EntryBox.delete("0.0",END)

#     if msg!='':
#         chatLog.config(state=NORMAL)
#         chatLog.insert(END, "You: "+msg+ '\n\n')
#         chatLog.config(foreground="#442265", font=("Verdana", 12 ))

#         res=chatbot_response(msg)
#         chatLog.insert(END, "Bot: "+ res +'\n\n')

#         chatLog.config(state=DISABLED)
#         chatLog.yview(END)



# base = Tk()
# base.title("Assisstant")
# base.geometry("400x500")
# base.resizable(width=False,height=False)

# chatLog = Text(base, bg="white", height="10", width="60", font="Arial")

# chatLog.config(state=DISABLED)

# scrollBar = Scrollbar(base, command=chatLog.yview, cursor="arrow")
# chatLog['yscrollcommand'] = scrollBar.set

# sendButton = Button(base, font=('Verdana',12,'bold'), text="Send", width="10", height="4", bd=0, bg="#32de97", activebackground="#3c9d9b", foreground="#ffffff", command=send)
# EntryBox = Text(base, bd=0, bg="white" ,width="25", height="5", font="Arial")


# #Place all components on the screen
# scrollBar.place(x=376,y=6, height=386)
# chatLog.place(x=6,y=6, height=386, width=370)
# EntryBox.place(x=128, y=401, height=90, width=265)
# sendButton.place(x=6, y=401, height=90)
# base.mainloop()




