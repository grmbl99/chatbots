import streamlit as st
from openai import OpenAI
from cosine_similarity import (
  process_text_files, 
  generate_embeddings, 
  save_embeddings, 
  load_embeddings, 
  find_similarities, 
  directory, 
  embeddings_filename
)

if __name__ == "__main__":
   
  client = OpenAI(
      base_url = 'http://localhost:11434/v1',
      api_key='ollama'
  )

  system_message = {"role": "system", "content": "You are a helpful assistant. You will only answer questions based on the context provided. You will always include a reference to context in which you found your answer. If you don't know the answer, say 'I don't know'."}

  st.title("Chatbot")
  st.write("This is a chatbot application using OpenAI's API with RAG (Retrieval-Augmented Generation).")
  st.write("You can ask me anything and I will try to respond as best as I can based on the provided context.")

  st.sidebar.title("Settings")

  if st.sidebar.button("Create new embeddings"):
      with st.spinner("Generating embeddings...", show_time=True):
          chunks = process_text_files(directory, chunk_size=25)
          embeddings_df = generate_embeddings(chunks)
          save_embeddings(embeddings_df, embeddings_filename)
          st.sidebar.success(f"Embeddings saved to {embeddings_filename}")

  temperature = st.sidebar.slider(
      'LLM Temperature',
      0.0, 1.0, 0.7
  )

  if st.sidebar.button("Clear chat history"):
      st.session_state.messages = []

  embeddings_df = load_embeddings(embeddings_filename)

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

    sorted_embeddings_df = find_similarities(prompt, embeddings_df)

    # use top 5 most similar chunks as context
    top_chunks = sorted_embeddings_df.head(5)
    context = "\n\n".join(
        f"Filename: {row['filename']}, Chunk ID: {row['chunk_id']}\n{row['chunk_text']}"
        for _, row in top_chunks.iterrows()
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Construct the full prompt with context
    full_prompt = f"""
    Context:
    {context}

    User Query:
    {prompt}
    """

    with st.spinner("Thinking..."):
      response = client.chat.completions.create(
        model="llama3.1:8b",
        temperature=temperature,
        max_tokens=4000,
        messages=[system_message] +  [{"role": "user", "content": full_prompt}]
      )

    with st.chat_message("assistant"):
      st.markdown(response.choices[0].message.content)

    # Show context references
    references = "\n".join(
        f"- Filename: {row['filename']}, Chunk ID: {row['chunk_id']}, Text: \"...{row['chunk_text']}...\""
        for _, row in top_chunks.iterrows()
    )
    st.markdown(f"**References:**\n{references}")

    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
