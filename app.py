import streamlit as st 
import pickle
import nltk
import string
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

tf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title(":blue[EMAIL SPAM DETECTION ] :man_technologist:")

sms = st.text_input("Enter the message ")
if(st.button("Predict")):
  #1. Preprocessing Message
  def transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
      if i.isalnum():
        y.append(i)

    text=y[:]
    y.clear()

    for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
        y.append(ps.stem(i))

    return " ".join(y)

  transformed_sms=transform(sms)

# 2. Vectorize the Message

  vector_sms=tf.transform([transformed_sms])

# 3. Model Prediction

  predited_sms=model.predict(vector_sms)[0]

# 4.Output

  if predited_sms==1:
    st.header("THIS EMAIL IS -> SPAM ")
    st.image("images.jpeg")
  else :
    st.header("THIS EMAIL IS -> NOT SPAM")
    st.image("not-spam.jpg")


