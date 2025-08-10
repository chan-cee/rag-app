import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA

# session = boto3.Session(
#     aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
#     aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
#     aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
#     region_name=st.secrets["AWS_DEFAULT_REGION"]
# )
session = boto3.Session()

bedrock = session.client("bedrock-runtime", region_name="us-west-2")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def get_vector_store(docs):
    # convert to embeddings
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings # titan model
    )
    vectorstore_faiss.save_local("faiss_index")

from langchain.schema import Document

def data_ingestion():
    """Split Excel data by 'Test Number' delimiter"""
    documents = []
    data_dir = "data"
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.xlsx', '.xls', '.csv')):
            file_path = os.path.join(data_dir, filename)
            
            try:
                # Read Excel/CSV file
                if filename.lower().endswith('.csv'):
                    sheets = {"CSV": pd.read_csv(file_path)}
                else:
                    sheets = pd.read_excel(file_path, sheet_name=None)
                
                for sheet_name, df in sheets.items():
                    if df.empty:
                        continue
                    
                    # Split by 'Test Number' delimiter
                    test_chunks = split_by_test_number(df, sheet_name, filename)
                    documents.extend(test_chunks)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    return documents

def split_by_test_number(df, sheet_name, filename):
    """Split DataFrame by rows containing 'Test Number', including initial content"""
    chunks = []
    
    # Find all rows that contain 'Test Number' in any column
    test_number_rows = []
    
    for idx, row in df.iterrows():
        # Check if any cell in the row contains 'Test Number'
        row_contains_test = any(
            str(cell).strip().startswith('TestNumber') 
            for cell in row.values 
            if pd.notna(cell)
        )
        
        if row_contains_test:
            test_number_rows.append(idx)
    
    # If no 'Test Number' found, treat entire sheet as one chunk
    if not test_number_rows:
        content = convert_entire_sheet_to_text(df, sheet_name, filename)
        chunk = Document(
            page_content=content,
            metadata={
                "source": filename,
                "sheet": sheet_name,
                "test_number": "No Test Number Found",
                "chunk_type": "full_sheet",
                "start_row": 1,
                "end_row": len(df)
            }
        )
        return [chunk]
    
    # Handle content BEFORE first Test Number (if any)
    if test_number_rows[0] > 0:
        # There's content before the first Test Number
        initial_chunk_df = df.iloc[0:test_number_rows[0]]
        
        content = convert_test_chunk_to_text(
            initial_chunk_df, sheet_name, filename, 
            "Header/Preamble", 1, test_number_rows[0]
        )
        
        chunk = Document(
            page_content=content,
            metadata={
                "source": filename,
                "sheet": sheet_name,
                "test_number": "Header/Preamble",
                "chunk_type": "initial_content",
                "start_row": 1,
                "end_row": test_number_rows[0],
                "total_rows": len(initial_chunk_df)
            }
        )
        chunks.append(chunk)
    
    # Create chunks between Test Number markers
    for i, start_row in enumerate(test_number_rows):
        # Determine end row (next Test Number or end of sheet)
        if i < len(test_number_rows) - 1:
            end_row = test_number_rows[i + 1]
        else:
            end_row = len(df)
        
        # Extract chunk data
        chunk_df = df.iloc[start_row:end_row]
        
        # Get test number from the first row
        test_number = extract_test_number(chunk_df.iloc[0])
        
        # Convert chunk to text
        content = convert_test_chunk_to_text(chunk_df, sheet_name, filename, test_number, start_row + 1, end_row)
        
        chunk = Document(
            page_content=content,
            metadata={
                "source": filename,
                "sheet": sheet_name,
                "test_number": test_number,
                "chunk_type": "test_based",
                "start_row": start_row + 1,
                "end_row": end_row,
                "total_rows": len(chunk_df)
            }
        )
        chunks.append(chunk)
    
    return chunks

def extract_test_number(first_row):
    """Extract the test number from the first row"""
    for cell in first_row.values:
        if pd.notna(cell) and 'Test Number' in str(cell):
            # Try to extract the number part
            cell_str = str(cell).strip()
            # Examples: "Test Number 1", "Test Number: 2", "Test Number 3.1"
            import re
            match = re.search(r'Test Number[:\s]*([0-9.]+)', cell_str, re.IGNORECASE)
            if match:
                return f"Test Number {match.group(1)}"
            else:
                return cell_str
    return "Unknown Test"

def convert_test_chunk_to_text(chunk_df, sheet_name, filename, test_number, start_row, end_row):
    """Convert a test chunk to structured text"""
    content_parts = [
        f"File: {filename}",
        f"Sheet: {sheet_name}",
        f"Test: {test_number}",
        f"Rows: {start_row}-{end_row}",
        f"Columns: {', '.join(chunk_df.columns)}",
        "=" * 60
    ]
    
    for idx, (original_idx, row) in enumerate(chunk_df.iterrows(), start=start_row):
        row_parts = []
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                row_parts.append(f"{col}: {value}")
        
        if row_parts:
            content_parts.append(f"Row {original_idx + 1}: {' | '.join(row_parts)}")
    
    return "\n".join(content_parts)

def convert_entire_sheet_to_text(df, sheet_name, filename):
    """Convert entire sheet when no Test Number found"""
    content_parts = [
        f"File: {filename}",
        f"Sheet: {sheet_name}",
        f"Total Rows: {len(df)}",
        f"Columns: {', '.join(df.columns)}",
        "=" * 60
    ]
    
    for idx, row in df.iterrows():
        row_parts = []
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                row_parts.append(f"{col}: {value}")
        
        if row_parts:
            content_parts.append(f"Row {idx + 1}: {' | '.join(row_parts)}")
    
    return "\n".join(content_parts)


system_template = (
    "You are an expert in semiconductor test automation and script translation. "
    "Your task is to convert legacy MCT 2000 test scripts into the modern SPEAL format. "
    "You must maintain the logic, structure, and intent of the original script while rewriting it using SPEAL syntax."
    "\nUse the following context as reference if helpful. Code on the LEFT represents SPEAL formatting, while code on the RIGHT represents MCT 2000 formatting.\n"
)

human_template = """
<context>
{context}
</context>

Question: {question}
"""

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template),
])

from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional, Dict
import json
from pydantic import Field

# custom wrapper
class GPTLLM(BaseLLM): # oss-120
    bedrock: Any = Field(exclude=True) 
    model_id: str = "openai.gpt-oss-120b-1:0"
    #model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    #model_id: str = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            try:
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.3),
                        "top_p": kwargs.get("top_p", 0.9),
                        "max_completion_tokens": kwargs.get("max_completion_tokens", 1024)
                    })
                )
                response_body = json.loads(response['body'].read())
                text = response_body['choices'][0]['message']['content']
                generations.append([Generation(text=text)])
            except Exception as e:
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "gpt-oss"
    
class ClaudeLLM(BaseLLM): # sonnet 4 
    bedrock: Any = Field(exclude=True) 
    #model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    model_id: str = "arn:aws:bedrock:us-west-2:897189464960:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            try:
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.3),
                        "top_p": kwargs.get("top_p", 0.9),
                        "max_tokens": kwargs.get("max_tokens", 1024)
                    })
                )
                response_body = json.loads(response['body'].read())
                text = response_body['content'][0]['text']
                generations.append([Generation(text=text)])
            except Exception as e:
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "claude"

# def get_response_llm(vectorstore_faiss, query):
#     session = boto3.Session(profile_name="Caleb")
#     bedrock = session.client("bedrock-runtime", region_name="us-west-2")
    
#     llm = GPTLLM(bedrock=bedrock) 
    
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore_faiss.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
#     return qa.invoke({"query": query})["result"]

def get_response_llm(vectorstore_faiss, query, model_id):
    #session = boto3.Session()
    bedrock = session.client("bedrock-runtime", region_name="us-west-2")

    if "openai" in model_id.lower():  
        llm = GPTLLM(bedrock=bedrock)
    elif "anthropic" in model_id.lower():
        llm = ClaudeLLM(bedrock=bedrock)
    else:
        raise ValueError(f"Unsupported model: {model_id}")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa.invoke({"query": query})["result"]

def display_top_k_chunks(vectorstore, query, k=3):
    top_k_docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    #st.markdown(f"## Top {k} Retrieved Chunks with Similarity Scores")

    for i, (doc, score) in enumerate(top_k_docs_with_scores, start=1):
        st.markdown(f"### Top Similarity Chunk: Chunk {i}")
        st.markdown(f"**Similarity Score:** {score:.4f}")
        st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
        st.markdown(f"**Sheet:** {doc.metadata.get('sheet', 'N/A')}")
        st.markdown(f"**Test Number:** {doc.metadata.get('test_number', 'N/A')}")
        st.markdown(f"**Rows:** {doc.metadata.get('start_row', 'N/A')} - {doc.metadata.get('end_row', 'N/A')}")
        
        with st.expander("📄 View Chunk Content"):
            st.text(doc.page_content[:3000]) # truncated


def main():
    st.set_page_config("RAG Application")
    
    st.header("🤖 Code Converter 🤖 ") # \n I am currently running using: OpenAI gpt-oss-120b

    user_question = st.text_input("Hello there! Ask a question in relation to the files:")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Update Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                print(f"Loaded and split {len(docs)} document chunks")
                get_vector_store(docs)
                st.success("Done")

    model_choice = st.selectbox("Choose LLM:", ["openai.gpt-oss-120b-1:0", "anthropic.claude-sonnet-4-20250514-v1:0"])

    if st.button("Get Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            
            #llm=get_claude_llm()
            #get_response_llm(faiss_index,user_question)
            st.markdown("## 🤓 LLM Answer")
            
            #st.write(get_response_llm(faiss_index,user_question))
            st.write(get_response_llm(faiss_index, user_question, model_choice))

            display_top_k_chunks(faiss_index, user_question, k=1)
            st.success("Done")

if __name__ == "__main__":
    main()