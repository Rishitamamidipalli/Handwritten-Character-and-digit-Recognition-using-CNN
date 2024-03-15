import pickle
import streamlit as st
import cv2
import numpy as np
import time

d={26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',0:'A',1:'B',1:'C',3:'D',4:'E',5:'F',6:'G',7:'H'
   ,8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

model=pickle.load(open('CNN_digit_mode.pkl', 'rb'))
st.header("Handwritten Digit and Charecter Recognization using CNN")
uploaded_file=st.file_uploader("Upload any Digit or Charecter Image of 28*28 pixel resolution")
if uploaded_file:
    
    file_name = uploaded_file.name
    print(file_name)
    
x=0
def read_image(uploaded_file):
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    pic = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # file_name = uploaded_file.name
    # pic=cv2.imread(file_name)[:,:,0]
    pic=np.invert(np.array([pic]))
    pre=model.predict(pic)
    return np.argmax(pre)
c1,c2=st.columns([7,3])
with c1:
     pass
with c2:
    
    if st.button("Show Predicted value"):
        x=1
        with st.spinner("LOADING....."):
            time.sleep(5)
if x==1:
    # c=st.container(height=50,border=True)
    y=read_image(uploaded_file)
    k=d[y]
    # c.write(f'The Predicted value is {k}')
    st.markdown(f'The Predicted value is {k}')