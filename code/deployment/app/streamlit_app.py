import streamlit as st
import requests

FASTAPI_URL =  "http://fastapi_app:8000/predict"

st.title("Doctor Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your sickness symptoms here"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    input_data = {"input_data": prompt}

    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()
    response = f"I guess your disease: {prediction['prediction']}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
