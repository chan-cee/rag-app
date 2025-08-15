import json
import os
import boto3
import pandas as pd
import numpy as np
import re
import streamlit as st
from modules import prompt, chunking, models

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

session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)
#session = boto3.Session() # for local running when logged into aws sso login

bedrock = session.client("bedrock-runtime", region_name="us-west-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock) # titan model

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

s3_client = session.client("s3")

PROMPT = prompt.get_prompt()

# def get_response_llm(vectorstore, query, model_id):
#     #session = boto3.Session()
#     bedrock = session.client("bedrock-runtime", region_name="us-west-2")

#     if "openai" in model_id.lower():  
#         llm = models.GPTLLM(bedrock=bedrock)
#     elif "anthropic" in model_id.lower():
#         llm = models.ClaudeLLM(bedrock=bedrock)
#     else:
#         raise ValueError(f"Unsupported model: {model_id}")
    
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff", # map_reduce , refine, stuff
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
#     output = qa.invoke({"query": query})["result"]
#     output = re.sub(r"<reasoning>.*</reasoning>", "", output, flags=re.DOTALL) # omit reasoning portion
#     return output

def get_response_llm(vectorstore, query, model_id):
    import re
    from langchain.prompts import PromptTemplate

    # Initialize Bedrock client
    bedrock = session.client("bedrock-runtime", region_name="us-west-2")

    # Select model wrapper
    if "openai" in model_id.lower():  
        llm = models.GPTLLM(bedrock=bedrock)
    elif "anthropic" in model_id.lower():
        llm = models.ClaudeLLM(bedrock=bedrock)
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    # ---------- Stage 1: Reformulate Query ----------
    reform_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=(
            "You are an expert query reformulator. "
            "Rewrite the user's query to make it more precise and relevant for document retrieval.\n"
            "Based on the test type in the MCT script, this rewritten query will be passed into the RAG pipeline.\n"
            "Currently the multiple SPEAL chunks in the corresponding to this test type have text starting with 'SPEAL #test_type Script Example #id_number'.\n" 
            "Those with the same id_number of the same test type are the relevant documents to retrieve to show the example of the conversion from MCT to SPEAL.\n"
            "The corresponding MCT code example begins with 'MCT2000 #test_type# Script Example #id_number', matching the same id_number as the SPEAL chunks. \n"
            "Do reformat this query below, in exact text literal form (can be fed word for word into the RAG for semantic search), in a way that can retrieve all those relevant example chunks to learn the conversion patterns.\n"
            "Original query: {original_query}\n"
            "Reformulated query:"
        )
    )

    reformulated_query = llm.invoke(
        reform_prompt.format(original_query=query)
    ).strip()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    output = qa.invoke({"query": reformulated_query})["result"]
    output = re.sub(r"<reasoning>.*?</reasoning>", "", output, flags=re.DOTALL)

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
        #st.markdown(f"**Test Number:** {doc.metadata.get('test_number', 'N/A')}")
        st.markdown(f"**Rows:** {doc.metadata.get('start_row', 'N/A')} - {doc.metadata.get('end_row', 'N/A')}")
        
        with st.expander("📄 View Chunk Content"):
            st.text(doc.page_content[:1500]) # truncated


def main():
    st.set_page_config("RAG Application")
    st.header("🧩  Code Refactoring Chatbot ") # \n I am currently running using: OpenAI gpt-oss-120b

    # user_input = st.text_input("Hello there! Ask a question in relation to the files:")
    # with st.sidebar:
    #     st.title("Update Or Create Vector Store:")
    #     uploaded_file = st.file_uploader("Upload your document", type=["xlsx", "csv", "pdf"])
    #     if uploaded_file is not None:
    #         s3_key = chunking.upload_file_to_s3(uploaded_file)
    #         st.success(f"Uploaded to S3 with key: {s3_key}")
        
    #     if st.button("Update Vectors"):
    #         with st.spinner("Processing..."):
    #             docs = chunking.data_ingestion()
    #             print(f"Loaded and split {len(docs)} document chunks")
    #             chunking.get_vector_store(docs, bedrock_embeddings)
    #             st.success("Done")

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
        #user_input = st.text_input("Hello there! Ask a question relevant to our code knowledge base:")
        user_input = st.text_area("Hello there! Ask any questions relevant to our code knowledge base:", height=75)
        model_choice = st.selectbox("LLM Model Selection:", ["openai.gpt-oss-120b-1:0", "anthropic.claude-sonnet-4-20250514-v1:0"])
        submitted = st.form_submit_button("Get Output")

    #if st.button("Get Output"):
    if submitted:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown("### 💭 LLM Answer")
        with st.spinner("Thinking..."):
            #vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)\
            #vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=bedrock_embeddings.embed_query)
            vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, pinecone_api_key=PINECONE_API_KEY, embedding=bedrock_embeddings)

            bot_response = get_response_llm(vectorstore, user_input, model_choice)
            #st.write(bot_response)
            st.info(bot_response)
            st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

            st.write("") 
            display_top_k_chunks(vectorstore, user_input, k=5)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("### Chat History")
            st.write("") 
            #for chat in st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {chat['user']}\n")
                st.markdown(f"**Bot:** {chat['bot']}")
                st.markdown("---")

            st.success("Done")

if __name__ == "__main__":
    main()