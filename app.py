import streamlit as st

st.title("Hello Streamlit App ")

st.write("Writing to streamlit app is working fine.")

name = st.text_input("Enter your name: ")

if name:
    st.success(f"Hello {name} !")