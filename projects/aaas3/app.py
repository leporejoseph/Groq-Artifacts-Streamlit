
import streamlit as st
from datetime import datetime

def main():
    st.title("Welcome to the new world")
    st.write("This is a basic Streamlit app template.")
    st.write(datetime.now())
if __name__ == "__main__":
    main()
