
import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from sas7bdat import SAS7BDAT
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Set your SAS file path here
SAS_FILE_PATH = "adae.sas7bdat"  # Replace with your SAS file path

@st.cache_resource
def initialize_rag_system():
    # Load SAS dataset
    with SAS7BDAT(SAS_FILE_PATH, skip_header=False) as reader:
        df = reader.to_data_frame()
    
    # Convert dataframe to text chunks
    text_data = df.to_string()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_data)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Setup RAG pipeline
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return df, qa_chain

# Initialize Streamlit app
st.title("SAS Dataset Q&A System")

try:
    # Initialize RAG system (this will only run once due to caching)
    df, qa_chain = initialize_rag_system()
    
    # Display basic dataset info
    st.write(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Question input
    user_question = st.text_input("Ask a question about your dataset:", 
                                 placeholder="e.g., What is the average value of column X?")
    
    if user_question:
        with st.spinner("Generating answer..."):
            # Get response from RAG
            response = qa_chain({"query": user_question})
            
            # Display answer
            st.markdown("### Answer")
            st.write(response["result"])
            
            # Display sources if needed
            with st.expander("View source data"):
                for doc in response["source_documents"]:
                    st.text(doc.page_content[:200] + "...")

except Exception as e:
    st.error(f"Error: Make sure your SAS file path is correct and OpenAI API key is set. \nDetails: {str(e)}")

# Add minimal instructions
st.sidebar.markdown("""
### Instructions
1. Set your SAS file path in the script
2. Ensure your OpenAI API key is in `.env`
3. Ask questions about your data in natural language

Example questions:
- What is the distribution of values in column X?
- How many unique values are in column Y?
- What is the relationship between columns A and B?
""")

# #-------------------------------fourth method-------------------------------------------------------------

# # Load dataset
# # try:
# #     data = pd.read_sas('skinproduct_attributes_seg.sas7bdat', format='sas7bdat')
# # except FileNotFoundError:
# #     st.error("Dataset file not found. Please check the file path.")
# #     st.stop()

# # Load dataset
# try:
#     data = pd.read_csv('Product_v6.csv')
# except FileNotFoundError:
#     st.error("Dataset file not found. Please check the file path.")
#     st.stop()

# # Example of creating a combined context from multiple fields
# # Combine relevant fields into a single context
# data['combined_context'] = (
#     data['name'].astype(str) + " " + 
#     data['description'].astype(str) + " " + 
#     data['value'].astype(str) + " " + 
#     data['status'].astype(str) + " " +
#     data['productType'].astype(str) + " " +
#     data['category.code'].astype(str)
# )

# # Initialize the question-answering pipeline
# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# # Create a TF-IDF Vectorizer for retrieval
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(data['combined_context'])

# # Streamlit UI
# st.title("SAS Dataset Question Answering")

# # Create a text input for the user to ask questions
# user_question = st.text_input("Ask a question about the dataset:")

# if st.button("Get Answer"):
#     if user_question:
#         # Step 1: Retrieve relevant entries based on the question
#         question_vector = vectorizer.transform([user_question])
#         similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
#         relevant_indices = similarities.argsort()[-5:][::-1]  # Top 5 similar items

#         # Gather relevant context from the top entries
#         context = ' '.join(data['combined_context'].iloc[relevant_indices])  # Using combined context
        
#         # Step 2: Use the QA pipeline to generate an answer
#         result = qa_pipeline(question=user_question, context=context)
        
#         # Display the answer
#         st.write("Answer:", result['answer'])
#     else:
#         st.write("Please enter a question.")


#------------------------------------------Third method-------------------------------------------------------------
# # Initialize OpenAI API
# openai.api_key = ''

# # Function to query OpenAI and get an answer
# def get_openai_response(question):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": question}]
#     )
#     return response['choices'][0]['message']['content']

# # Streamlit UI
# st.title("SAS Dataset Question Answering")
# user_question = st.text_input("Ask a question about the dataset:")

# if st.button("Get Answer"):
#     if user_question:
#         # Get response from OpenAI
#         openai_response = get_openai_response(user_question)
        
#         # (Optional) Here, you can also process the response to query your dataset if needed
        
#         st.write("OpenAI's Response:", openai_response)
#     else:
#         st.write("Please enter a question.")


# #------------------------------------------Second try---------------------------------------------------------------
# # Field descriptions mapping
# field_descriptions = {
#     "ProductName": "The name of the product.",
#     "Price": "The price of the product in USD.",
#     "Category": "The category under which the product is classified.",
#     "Brand": "The brand name of the product.",
#     "Rating": "The average customer rating for the product on a scale from 1 to 5.",
#     # Add more fields as necessary
# }

# # Display dataset info and descriptions
# st.write("Dataset Overview:")
# st.write(data.head())

# st.write("Field Descriptions:")
# for field, description in field_descriptions.items():
#     st.write(f"**{field}**: {description}")

# # Initialize BERT question-answering pipeline
# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# # Function to analyze query and provide response
# def analyze_query(query):
#     # Combine field descriptions into context
#     descriptions = "\n".join([f"{field}: {desc}" for field, desc in field_descriptions.items()])
#     context = f"Dataset field descriptions:\n{descriptions}\n\nDataset contents:\n{data.to_string(index=False)}"

#     # Check for basic statistics requests
#     if "average" in query.lower() or "mean" in query.lower():
#         numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
#         if numeric_cols:
#             means = data[numeric_cols].mean()
#             return f"Average values:\n{means.to_string()}"
#         else:
#             return "No numeric columns to calculate averages."

#     elif "count" in query.lower() or "number of" in query.lower():
#         categorical_counts = data.select_dtypes(include='object').nunique()
#         return f"Unique counts per categorical column:\n{categorical_counts.to_string()}"

#     elif "most common" in query.lower() or "top" in query.lower():
#         most_common = {}
#         for col in data.select_dtypes(include='object').columns:
#             most_common[col] = data[col].value_counts().idxmax()
#         return f"Most common values:\n{most_common}"

#     else:
#         # Use QA model for complex queries
#         answer = qa_pipeline({'question': query, 'context': context})
#         return answer['answer'] if answer['score'] > 0.1 else "I'm not sure about that."

# # Streamlit user interface
# st.title("Natural Language Query App for SAS Dataset")
# query = st.text_input("Ask a question about the dataset:")

# if query:
#     answer = analyze_query(query)
#     st.write("Answer:", answer)


#----------------------------------------First try--------------------------------------------------------------
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


