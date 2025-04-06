import streamlit as st
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama'
)

system_message = {"role": "system", "content": "You are an army general and will answer in a military tone."}

st.title("Chatbot")
st.write("This is a simple chatbot application using OpenAI's API.")
st.write("You can ask me anything and I will try to respond as best as I can.")

st.sidebar.title("Settings")

temperature = st.sidebar.slider(
    'LLM Temperature',
    0.0, 1.0, 0.7
)

if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
  with st.chat_message("user"):
    st.markdown(prompt)

  st.session_state.messages.append({"role": "user", "content": prompt})

  with st.spinner("Thinking..."):
    response = client.chat.completions.create(
      model="llama3.1:8b",
      temperature=temperature,
      max_tokens=4000,
      messages=[system_message] + st.session_state.messages
    )

  with st.chat_message("assistant"):
    st.markdown(response.choices[0].message.content)

  st.session_state.messages.append(
     {"role": "assistant", "content": response.choices[0].message.content})
