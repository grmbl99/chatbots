import streamlit as st
import pandas as pd
import numpy as np
 
st.write("""
# My first app
Hello *world!*
""")

add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)
st.write('You selected:', add_selectbox)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)
st.write('Values:', add_slider)

number = st.slider("Pick a number", 0, 100)
st.write("The current number is ", number)

date = st.date_input("Pick a date", pd.to_datetime("2023-01-01"))
st.write("The current date is ", date)

pet = st.selectbox("Pick a pet", ["Dog", "Cat", "Fish"])
st.write("The current pet is ", pet)

st.text_input("Your name", key="name")
st.write("The current name is ", st.session_state.name)

df = pd.read_csv("my_data.csv")
st.line_chart(df)

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)