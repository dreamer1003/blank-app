import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import faiss

# Load dataset
# try:
#     data = pd.read_sas('skinproduct_attributes_seg.sas7bdat', format='sas7bdat')
# except FileNotFoundError:
#     st.error("Dataset file not found. Please check the file path.")
#     st.stop()

# Load dataset
try:
    data = pd.read_csv('Product_v6.csv')
except FileNotFoundError:
    st.error("Dataset file not found. Please check the file path.")
    st.stop()

# Display dataset info
st.write("Dataset Overview:")
st.write(data.head())

# Initialize BERT question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to analyze query and provide response
def analyze_query(query):
    # Check for basic statistics requests
    if "average" in query.lower() or "mean" in query.lower():
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            means = data[numeric_cols].mean()
            return f"Average values:\n{means.to_string()}"
        else:
            return "No numeric columns to calculate averages."

    elif "count" in query.lower() or "number of" in query.lower():
        categorical_counts = data.select_dtypes(include='object').nunique()
        return f"Unique counts per categorical column:\n{categorical_counts.to_string()}"

    elif "most common" in query.lower() or "top" in query.lower():
        most_common = {}
        for col in data.select_dtypes(include='object').columns:
            most_common[col] = data[col].value_counts().idxmax()
        return f"Most common values:\n{most_common}"

    else:
        # Use QA model for complex queries
        context = data.to_string(index=False)
        answer = qa_pipeline({'question': query, 'context': context})
        return answer['answer'] if answer['score'] > 0.1 else "I'm not sure about that."

# Streamlit user interface
st.title("Natural Language Query App for SAS Dataset")
query = st.text_input("Ask a question about the dataset:")

if query:
    answer = analyze_query(query)
    st.write("Answer:", answer)









# # Display dataset info
# st.write("Dataset Overview:")
# st.write(data.head())

# # Load a model for sentence embeddings
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to extract statistics based on user queries
# def extract_statistics(column_name):
#     if pd.api.types.is_numeric_dtype(data[column_name]):
#         mean_value = data[column_name].mean()
#         median_value = data[column_name].median()
#         min_value = data[column_name].min()
#         max_value = data[column_name].max()
#         return (f"The average of '{column_name}' is {mean_value:.2f}, "
#                 f"the median is {median_value:.2f}, "
#                 f"the minimum is {min_value:.2f}, "
#                 f"and the maximum is {max_value:.2f}.")
#     elif pd.api.types.is_categorical_dtype(data[column_name]) or pd.api.types.is_object_dtype(data[column_name]):
#         value_counts = data[column_name].value_counts().nlargest(5)
#         counts_str = ', '.join([f"{index}: {value}" for index, value in value_counts.items()])
#         return f"The most common values in '{column_name}' are: {counts_str}."
#     else:
#         return "This column type is not supported for analysis."

# # Function to interpret the user query
# def get_answer(query):
#     # Create embeddings for the column names
#     column_names = data.columns.tolist()
#     column_embeddings = embedder.encode(column_names, convert_to_tensor=True)

#     # Generate embedding for the user's question
#     query_embedding = embedder.encode(query, convert_to_tensor=True)

#     # Calculate cosine similarity between query and column names
#     cosine_scores = util.pytorch_cos_sim(query_embedding, column_embeddings)[0]

#     # Get the best matching column based on cosine similarity
#     best_match_idx = cosine_scores.argmax().item()
#     best_column = column_names[best_match_idx]

#     # Retrieve statistics based on the matched column
#     response = extract_statistics(best_column)

#     return response

# # Streamlit interface for user query
# st.title("Natural Language Query App")
# query = st.text_input("Ask a question about the dataset:")

# if query:
#     answer = get_answer(query)
#     st.write(answer)