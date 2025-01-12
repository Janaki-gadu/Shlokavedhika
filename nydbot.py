import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from sentence_transformers.util import cos_sim
import json

# Load the datasets
@st.cache_data
def load_datasets():
    gita_questions = pd.read_csv("gita.csv")[['question', 'translation', 'verse', 'sanskrit', 'chapter', 'speaker']].dropna()
    patanjali_questions = pd.read_csv("pys.csv")[['question', 'translation', 'verse', 'sanskrit','chapter']].dropna()
    gita_questions.columns = ['question', 'answer', 'verse', 'sanskrit', 'chapter', 'speaker']
    patanjali_questions.columns = ['question', 'answer', 'verse', 'sanskrit','chapter']
    return pd.concat([gita_questions, patanjali_questions], ignore_index=True)

# Load and process data
df_combined = load_datasets()

# Ensure necessary columns are strings
df_combined = df_combined.astype(str)

if df_combined.empty:
    st.error("Combined dataset is empty. Please check the input datasets.")
    st.stop()

# Initialize models
@st.cache_resource
def initialize_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    summarizer = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, summarizer

model, tokenizer, summarization_model = initialize_models()

# Create embeddings and FAISS index
@st.cache_resource
def create_index(df):
    question_embeddings = model.encode(df['question'].tolist(), convert_to_tensor=True)
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(question_embeddings.cpu().numpy())
    return index

index = create_index(df_combined)

# Function to retrieve answers with relevance scoring
def retrieve_and_filter_answers(query, k=5, relevance_threshold=None):
    query_embedding = model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), k)
    filtered_results = []
    
    for idx in indices[0]:
        answer_embedding = model.encode([df_combined.iloc[idx]['question']], convert_to_tensor=True)
        relevance_score = cos_sim(query_embedding, answer_embedding).item()
        
        # Use the provided relevance threshold or default to 0.7
        threshold = relevance_threshold if relevance_threshold else 0.7
        
        if relevance_score >= threshold:
            filtered_result = {
                "question": df_combined.iloc[idx]['question'],
                "answer": df_combined.iloc[idx]['answer'],
                "verse": df_combined.iloc[idx]['verse'],
                "sanskrit": df_combined.iloc[idx]['sanskrit'],
                "chapter": df_combined.iloc[idx].get('chapter', 'N/A'),
                "speaker": df_combined.iloc[idx].get('speaker', None),
                "relevance_score": relevance_score
            }
            
            # Only include speaker information if it's available
            if filtered_result['speaker'] is None:
                filtered_result.pop('speaker')  # Remove speaker if None
            
            filtered_results.append(filtered_result)

    return sorted(filtered_results, key=lambda x: x['relevance_score'], reverse=True)

def generate_summary(answers):
    combined_text = "\n".join([f"Answer: {ans['answer']}\nVerse: {ans['verse']}\nChapter: {ans.get('chapter', 'N/A')}\nSanskrit: {ans['sanskrit']}\n"
                               for ans in answers])
    prompt = f"Here are several answers to a user's query. Summarize them into a concise, coherent paragraph with key details.\n{combined_text}"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summarization_model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Sorry, I couldn't generate a summary at this moment. Please try again."


from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_6TYWqe_4rZoTuYciz9f8fSpBZsAYhJNnSKJZmbs19kiGueQ1e6fGVpXj7dvMGQAGfUzcqT")
index_name = "gita-patanjali-index"
environment = "us-east-1"

# Ensure the index exists with the correct dimension
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Ensure this matches your vector dimension
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region=environment)
    )

# Define the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample DataFrame (replace with your actual dataframe)
df_combined = pd.DataFrame({
    'chapter': [1, 2],
    'verse': [1, 2],
    'sanskrit': ['धृतराष्ट्र उवाच |', 'भगवान उवाच |']
})

# Function to generate embedding
def generate_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Forward pass through the model to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings (typically the output of the [CLS] token, i.e., first token embedding)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average over all tokens
    
    return embeddings


# Prepare vectors to upsert into Pinecone
vectors = []
for _, row in df_combined.iterrows():
    # Ensure vector ID is ASCII-only
    raw_vector_id = f"{row['chapter']}-{row['verse']}-{row['sanskrit'][:10]}"  # Use first 10 characters of 'sanskrit'
    vector_id = raw_vector_id.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    
    # Generate embedding for the 'sanskrit' text
    embedding = generate_embedding(row['sanskrit'])
    
    # Add to vectors list for upsert
    vectors.append({"id": vector_id, "values": embedding.tolist()})  # Convert the embedding to list for compatibility

# Connect to Pinecone and upsert the vectors (example code for Pinecone connection)
index = pc.Index(index_name)
index.upsert(vectors)

# Function to add embeddings to Pinecone
def add_embeddings_to_pinecone(df):
    embeddings = generate_embedding(df['sanskrit'].tolist())  # Use the 'sanskrit' column instead of 'question'
    vectors = [(str(i), embedding.tolist()) for i, embedding in enumerate(embeddings)]
    index = pc.Index(index_name)  # Ensure the index is connected
    index.upsert(vectors)

add_embeddings_to_pinecone(df_combined)  # Add embeddings to Pinecone

# Function to retrieve relevant answers from Pinecone
def retrieve_from_pinecone(query, k=5):
    # Generate embedding for the query
    query_embedding = generate_embedding(query)  # Generate the embedding for the query
    
    # Ensure the embedding is a flat list (dimension of 768)
    query_embedding = query_embedding.tolist()  # Convert to list if it's a numpy array
    
    # Perform the query
    index = pc.Index(index_name)  # Ensure the index is connected
    results = index.query(vector=query_embedding, top_k=k)  # Query the Pinecone index
    return results['matches']

# Example retrieval
results = retrieve_from_pinecone("What is Yoga?", k=5)
for result in results:
    print(result)



# Streamlit UI
background_image_url = 'https://files.oaiusercontent.com/file-6fsxN3Y7BbWjc87u44a2MH?se=2025-01-11T14%3A11%3A31Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D6f9f5427-9e79-44ce-98ee-558c22017860.webp&sig=hbCaoCtrJebHwlJnSxpy4g5eKUpk1Frn/vAUpihtXGc%3D'

# Define custom CSS for background
st.markdown(
    f"""
    <style>
    .stApp {{
            background-image: url("{background_image_url}");
            background-size: 100% 100%; /* Stretch to fit screen */
            background-position: center;
            background-repeat: no-repeat;
            margin: 0;
            padding: 0;
          
        }}
        .reportview-container .main .block-container {{
            background: rgba(0, 0, 0, 0.5);  /* Add transparency to main content area */
            padding: 20px;
            border-radius: 10px;
        }}
        .sidebar .sidebar-content {{
            background: rgba(0, 0, 0, 0.7);  /* Sidebar with higher transparency */
        }}
        .stTextInput input {{
            background-color: rgba(255, 255, 255, 0.8);
            color: black;  /* For better contrast on white background */
        }}
        /* Add padding at the bottom of the main content to avoid overlap */
        .main {{
            padding-bottom: 80px; 
        }}
        /* Fix the input box at the bottom */
        .stTextInput {{
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%); 
            right: 0;
            z-index: 2;
            padding: 20px;
        }}
        /* Make sure the input box is responsive */
        @media (max-width: 768px) {{
            .stTextInput input {{
                font-size: 16px;  /* Adjust font size for small screens */
            }}
        }}
        /* Chat bubble styles */
        .user-message {{
            text-align: left;
            color: black;
            background-color: rgba(255, 253, 240,0.9);
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 100%;
            word-wrap: break-word;
            display: inline-block;
            font-weight:bold;
         
        }}
        .bot-message {{
            text-align: left;
            background-color: rgba(255, 253, 240,0.7);
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 100%;
            word-wrap: break-word;
            display: inline-block;
            
        }}
        .title {{
        position: fixed;
        top: 38px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 36px;
        font-weight: bold;
        color: #F5F5DC;
    }}
    .subtitle {{
        position: fixed;
        top: 80px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 18px;
        font-weight: normal;
        color: #F5F5DC;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Initialize session state for storing queries and answers
if 'queries' not in st.session_state:
    st.session_state.queries = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'query_processed' not in st.session_state:
    st.session_state.query_processed = False

# Function to handle query submission
def submit_query():
    if st.session_state.query_input.strip():
        results = retrieve_and_filter_answers(st.session_state.query_input.strip())
        if results:
            st.session_state.queries.append(st.session_state.query_input.strip())
            st.session_state.answers.append(results)
            summary = generate_summary(results)
            st.session_state.query_processed = True
            st.session_state.query_input = "" # Clear the input field
            st.session_state.summary = summary
        else:
            # Show top 3 most relevant answers (fallback behavior)
            st.write("No exact match found. Showing the top 3 most relevant answers:")
            results = retrieve_and_filter_answers(st.session_state.query_input.strip(), k=3)
            if results:
                st.session_state.queries.append(st.session_state.query_input.strip())
                st.session_state.answers.append(results)
                st.session_state.query_processed = True
            else:
                st.write("No relevant answers found. Please try again.")
    else:
        st.write("Please enter a query before submitting.")

# Display the title and description
st.markdown('<h1 class="title">SHLOKAVEDHIKA</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Unveiling Ancient Wisdom Through Modern Technology.</h2>', unsafe_allow_html=True)

# Display all previous queries and their answers in a conversational manner
if st.session_state.queries:
    for i, (query, answers) in enumerate(zip(st.session_state.queries, st.session_state.answers)):
        # Display user query
        st.markdown(f'<div class="user-message">{query}</div>', unsafe_allow_html=True)
        for result in answers:
            speaker_text = ""
            # Check if 'speaker' exists and is valid before adding it to the response
            if 'speaker' in result and result['speaker'] not in [None, 'nan', 'NaN', '']:
                speaker_text = f"<b>Speaker:</b> {result['speaker']}<br>"
            else:
                speaker_text = ""

            bot_response = f"""
            <b>Answer:</b> {result['answer']}<br>
            <b>Verse:</b> {result['verse']}<br>
            <b>Chapter:</b> {result['chapter']}<br>
            <b>Shloka:</b> {result['sanskrit']}<br>
            {speaker_text}
            """

# Use st.expander for collapsible verse details
            with st.expander(f"See more details for Verse {result['verse']}"):
                st.markdown(f"<div class='bot-message'>{bot_response}</div>", unsafe_allow_html=True)
# Create a fixed input at the bottom of the page using st.empty()
st.markdown('<div style="height:150px;position:fixed;"></div>', unsafe_allow_html=True)  # Spacer to avoid overlap
query_box = st.empty()  # Placeholder for the input box

# Text input for new queries
query_box.text_input("Unlock the insights of Knowledge:", key="query_input", placeholder="Ask a question...", label_visibility="collapsed", on_change=submit_query)
