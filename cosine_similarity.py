import os
import json
import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

directory = "./text"
embeddings_filename = "embeddings.txt"

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama')

def chunk_text(text, chunk_size):
    chunks = []
    words = text.split()  # Split by whitespace
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i: i+chunk_size]))
    return chunks

def process_text_files(directory, chunk_size):
    chunks=[]
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                file_chunks = chunk_text(text, chunk_size)
                for i in range(len(file_chunks)):
                    chunks.append({"filename": filename, "chunk_id": i, "chunk_text": file_chunks[i]})
    return chunks

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        text = chunk["chunk_text"]
        response = client.embeddings.create(
            model="nomic-embed-text",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    embeddings_df = pd.DataFrame(chunks)
    embeddings_df["embedding"] = embeddings
    return embeddings_df

def save_embeddings(embeddings, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        embeddings.to_csv(output_file, index=False)

def load_embeddings(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        embeddings_df = pd.read_csv(input_file)
    # Convert string representation of list to actual list
    embeddings_df["embedding"] = embeddings_df["embedding"].apply(json.loads)
    return embeddings_df

def find_similarities(input_text, embeddings_df):
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=input_text
    )
    input_embedding = response.data[0].embedding
    
    embeddings = embeddings_df["embedding"].tolist()
    similarities = cosine_similarity([input_embedding], embeddings)

    embeddings_df["similarity"] = similarities[0]
    sorted_embeddings_df = embeddings_df.sort_values(by="similarity", ascending=False)

    return sorted_embeddings_df

if __name__ == "__main__":

    st.title("Document Similarity Search")
    st.write("This application allows you to search for similar documents based on cosine similarity of embeddings.")

    st.sidebar.title("Settings")

    if st.sidebar.button("Create new embeddings"):
        with st.spinner("Generating embeddings...", show_time=True):
            chunks = process_text_files(directory, chunk_size=25)
            embeddings_df = generate_embeddings(chunks)
            save_embeddings(embeddings_df, embeddings_filename)
            st.sidebar.success(f"Embeddings saved to {embeddings_filename}")

    embeddings_df = load_embeddings(embeddings_filename)

    if searchtext := st.chat_input("Search text"):
        sorted_embeddings_df = find_similarities(searchtext, embeddings_df)
        most_similar = sorted_embeddings_df.iloc[0]

        st.subheader('Results:')
        st.dataframe(
            sorted_embeddings_df.style.highlight_max(subset=["similarity"], axis=0), 
            hide_index=True, 
            column_order=("filename","chunk_id","similarity","chunk_text") )

        st.subheader('Chunk:')
        st.write(most_similar["chunk_text"])

        #Show the content of the most similar document
        st.subheader('Document content:')
        most_similar_filepath = os.path.join(directory, most_similar["filename"])
        with open(most_similar_filepath, "r", encoding="utf-8") as file:
            most_similar_text = file.read()
        st.write(most_similar_text)