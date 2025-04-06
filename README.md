# Chatbots

This repository illustrates how easy it is to create a chatbot using a combination of the following technologies:

- Python 3.12
- Ollama
- Streamlit

## Ollama

Ollama is a command-line tool that allows you to run large language models locally. It provides a simple interface for downloading and running models.

Install Ollama by following the instructions on their [website](https://ollama.com/).

We use the following local models:

- llama3.1:8b
- nomic-embed-text

Install these models by running the following commands:

```bash
ollama run llama3.1:8b
ollama run nomic-embed-text
```

You can see which models you are running by executing the following command:

```bash
ollama list
```

Models are exposed as REST API's on localhost:11434. The scripts use the OpenAPI library to interact with the models.

Install the Python OpenAI API by running the following command:

```bash
pip install openai
```

## Streamlit

Streamlit is a Python library that allows you to create web applications for machine learning and data science projects. It provides a simple way to build interactive user interfaces.

Install Streamlit by running the following command:

```bash
pip install streamlit
```

To run a Streamlit application, navigate to the directory where your `app.py` file is located and run the following command:

```bash
streamlit run app.py
```

## Miscellaneous

The scripts uses the following libraries:

- pandas
- numpy
- sklearn

Install them as follows:

```bash
pip install pandas numpy scikit-learn
```

## Usage

To use the chatbot, run the Streamlit application as described above. The applications will provide a web user interface where you can interact with the chatbots.

There are 3 chatbots available:

### **chatbot.py**

Simple chatbot application using OpenAI's API.

```bash
streamlit run chatbot.py
```

### **cosine_similarity.py**

Application showing how to search documents based on cosine similarity of embeddings.

Some example documents are included in the `text` directory (the script only uses `*.txt` files in this directory).

The embeddings are stored in a local `embeddings.txt` file. When the script is run for the first time, press the "Generate Embeddings" button to generate this file.

```bash
streamlit run cosine_similarity.py
```

### **chatbot_rag.py**

Chatbot application using OpenAI's API with RAG (Retrieval-Augmented Generation). Combining 1 and 2.

This script imports the `cosine_similarity.py` script to re-use it's embeddings functions.

```bash
streamlit run chatbot_rag.py
```
