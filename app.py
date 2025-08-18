import json
import os
import boto3
import pandas as pd
import numpy as np
import re
import streamlit as st
from modules import prompt, chunking, models
from botocore.config import Config
import tiktoken

from langchain_community.embeddings import BedrockEmbeddings
#from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone as LangPinecone

PINECONE_INDEX_NAME = "rag-documents"  
S3_BUCKET_NAME = "rag-documents-eds"
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

config = Config(
    read_timeout=300,
    connect_timeout=30,
    retries={'max_attempts': 1} 
)

session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)
#session = boto3.Session() # for local running when logged into aws sso login

bedrock = session.client("bedrock-runtime", region_name="us-west-2", config=config)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock) # titan model

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

s3_client = session.client("s3")

PROMPT = prompt.get_prompt()

def get_response_llm_streaming(vectorstore, query, model_id, k=7):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)
    system_content = f"""
                    You are an expert in semiconductor test automation, with deep knowledge of both legacy MCT 2000 scripting and modern SPEAL syntax.
                    Your goal is to accurately convert MCT 2000 scripts into SPEAL, preserving the test logic, structure, and purpose exactly.
                    All converted code must be valid, executable SPEAL with correct syntax, indentation, and section ordering.
                    Context:
                    {context}
                    Output:
                    Return the converted SPEAL code inside a single code block. DO NOT add any explanations except possible comments within the code. 
                    """

    model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    model_id = "arn:aws:bedrock:us-west-2:897189464960:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    payload = {
        #"model": model_id,
        "anthropic_version": "bedrock-2023-05-31",
        # "messages": [
        #     {"role": "system", "content": system_content},
        #     {"role": "user", "content": query} 
        # ],
        "system": system_content,  # top-level system instructions
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": 20000,
        "temperature": 0.2,
    }

    response_stream = bedrock.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json"
    )

    enc = tiktoken.get_encoding("cl100k_base")
    token_placeholder = st.empty()
    placeholder = st.empty()
    output_text = ""
    token_count = 0

    for event in response_stream["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk.get("type") == "content_block_delta":
            delta_text = chunk["delta"].get("text", "")
            output_text += delta_text
            #st.write(delta_text, end="")  # Stream live
            placeholder.markdown(output_text)

            # approximate token count
            token_count += len(enc.encode(delta_text))
            token_placeholder.markdown(f"**Tokens generated so far:** {token_count}")

    return output_text


def get_response_llm(vectorstore, query, model_id, k=7):
    #bedrock = session.client("bedrock-runtime", region_name="us-west-2", config=config)

    if "openai" in model_id.lower():  
        llm = models.GPTLLM(bedrock=bedrock)
    elif "anthropic" in model_id.lower():
        llm = models.ClaudeLLM(bedrock=bedrock)
    else:
        raise ValueError(f"Unsupported model: {model_id}")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # map_reduce , refine, stuff
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    output = qa.invoke({"query": query})["result"]
    output = re.sub(r"<reasoning>.*</reasoning>", "", output, flags=re.DOTALL) # omit reasoning portion
    return output

def display_top_k_chunks(vectorstore, query, k=3):
    top_k_docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    st.markdown(f"### RAG - Most Relevant Chunks")
    st.write("") 
    for i, (doc, score) in enumerate(top_k_docs_with_scores, start=0):
        #st.markdown(f"#### Top Similarity Chunk: Chunk {i}")
        st.markdown(f"**Similarity Score:** {score:.4f}")
        st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
        st.markdown(f"**Sheet:** {doc.metadata.get('sheet', 'N/A')}")
        st.markdown(f"**Rows:** {doc.metadata.get('start_row', 'N/A')} - {doc.metadata.get('end_row', 'N/A')}")
        
        with st.expander("📄 View Chunk Content"):
            st.text(doc.page_content[:1500]) # truncated


def main():
    st.set_page_config("RAG Application")
    st.header("🧩  Code Refactoring Chatbot ") # \n I am currently running using: OpenAI gpt-oss-120b

    with st.sidebar:
        st.title("Vector Store Management")

        # --- Add single file ---
        uploaded_file = st.file_uploader("Upload document into Vector Store") #  (xlsx/csv/pdf)
        st.subheader("Chunking Strategy")
        chunking_type = st.radio(
            "Select chunking method:",
            options=["Token Count", "Test Number"],
            index=0,  # Default to Token Count
            help="Choose how to split your documents:\n"
                "• Token Count: Split by token limits with overlap\n"
                "• Test Number: Split by 'Test Number' markers in data"
        )

        if uploaded_file is not None and st.button("Add to Vector Store"):
            with st.spinner(f"Chunking and uploading {uploaded_file.name} to Pinecone..."):
                #chunking.upload_file_to_s3(uploaded_file) # upload to s3
                if chunking_type == "Token Count":
                    chunking.upload_chunks(
                        uploaded_file, 
                        bedrock_embeddings, 
                        "Token Count"
                    )
                else:  # Test Number
                    chunking.upload_chunks(
                        uploaded_file, 
                        bedrock_embeddings, 
                        "Test Number"
                    )
                #chunking.upload_chunks_test_based(uploaded_file, bedrock_embeddings)
                #chunking.upload_chunks(uploaded_file, bedrock_embeddings)
                st.success("Uploaded document to S3 and chunks to Pinecone!")

        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # --- Refresh entire vector store ---
        st.markdown("##### Refresh the entire vector store from AWS S3 Bucket")
        if st.button("Refresh Vector Store"):
            with st.spinner("Refreshing vector store from all S3 documents..."):
                keys = chunking.get_all_s3_keys("rag-documents-eds")
                total_chunks = 0
                for key in keys:
                    # chunk and upload each file to Pinecone
                    chunking.upload_chunks_from_s3(key, bedrock_embeddings)
                    st.write(f"Processed {key}")
                st.success("Vector store refreshed with all documents!")


    with st.form(key="input_form"):
        user_input = st.text_area("Hello there! Ask any questions relevant to our code knowledge base:", height=75)
        model_choice = st.selectbox("LLM Model Selection:", ["openai.gpt-oss-120b-1:0", "anthropic.claude-sonnet-4-20250514-v1:0", "anthropic.claude-3-7-sonnet-20250219-v1:0"])
        submitted = st.form_submit_button("Get Output")

    #if st.button("Get Output"):
    if submitted:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown("### 💭 LLM Answer")
        with st.spinner("Thinking..."):
            #vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)\
            vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, pinecone_api_key=PINECONE_API_KEY, embedding=bedrock_embeddings)
            if model_choice == "anthropic.claude-3-7-sonnet-20250219-v1:0":
                bot_response = get_response_llm_streaming(vectorstore, user_input, model_choice, 5)
                st.info(bot_response)
            else:
                bot_response = get_response_llm(vectorstore, user_input, model_choice, 5)
                st.info(bot_response)
                #st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

            st.write("") 
            display_top_k_chunks(vectorstore, user_input, k=5)
            st.markdown("<br>", unsafe_allow_html=True)

            # st.markdown("### Chat History")
            # st.write("") 
            # for chat in reversed(st.session_state.chat_history):
            #     st.markdown(f"**You:** {chat['user']}\n")
            #     st.markdown(f"**Bot:** {chat['bot']}")
            #     st.markdown("---")

            st.success("Done")

if __name__ == "__main__":
    main()