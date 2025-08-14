from langchain.schema import Document
import pandas as pd
import boto3
import streamlit as st
import io
import os
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import FAISS
import tiktoken
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter


S3_BUCKET_NAME = "rag-documents-eds"
PINECONE_INDEX_NAME = "rag-documents"
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

s3_client = session.client("s3")

# s3_client = boto3.client("s3",
#     aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
#     aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
#     aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
#     region_name=st.secrets["AWS_DEFAULT_REGION"]
# )

def upload_file_to_s3(uploaded_file) -> str: # all file types are accepted (might need to fix)
    key = f"raw_docs/{uploaded_file.name}"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=uploaded_file.read()
        )
        uploaded_file.seek(0) 
        return key
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        raise  # re-raise after logging to not hide the error

def download_file_from_s3(key: str, local_path: str):
    """Downloads a file from S3 to a local path."""
    s3_client.download_file(S3_BUCKET_NAME, key, local_path)
    return local_path

def get_all_s3_keys(bucket_name):
    """List all object keys in the S3 bucket (you may want to filter prefix 'raw_docs/')"""
    keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix="raw_docs/"):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])
    return keys

# main function for upload to s3 and pinecone
def upload_chunks(uploaded_file, bedrock_embeddings, chunking_method): # only excel and csv (pdf will not be added to pinecone yet)
    # Read bytes from uploaded file
    uploaded_file.seek(0) 
    file_bytes = uploaded_file.read()
    
    # Determine file type and load DataFrame(s)
    # if uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
    #     sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    # elif uploaded_file.type == "text/csv":
    #     sheets = {"CSV": pd.read_csv(io.BytesIO(file_bytes))}
    # else:
    #     raise ValueError("Unsupported file type. Only Excel or CSV allowed.")
    filename = uploaded_file.name.lower()
    if filename.endswith(('.xlsx', '.xls')):
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    elif filename.endswith('.csv'):
        sheets = {"CSV": pd.read_csv(io.BytesIO(file_bytes))}
    else:
        raise ValueError(f"Unsupported file extension: {filename}")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30000,
        chunk_overlap=200,
        length_function=len,
    )

    # If multiple sheets, iterate through each
    for sheet_name, df in sheets.items():
        print(f"Chunking up {sheet_name}")
        text = df.to_csv(index=False) 
        original_doc = Document(
            page_content=text,
            metadata={"source": uploaded_file.name, "sheet": sheet_name}
        )
        chunks = splitter.split_documents([original_doc])
        all_docs.extend(chunks)


        # if chunking_method == 'Token Count':
        #     chunks = split_by_tokens(df, sheet_name, uploaded_file.name)
        #     all_docs.extend(chunks)
        # else: # test number
        #     chunks = split_by_test_number(df, sheet_name, uploaded_file.name)
        #     all_docs.extend(chunks)

    st.success(f"{len(all_docs)} chunks created from {uploaded_file.name}! Uploading to Pinecone now...")

    vectorstore = PineconeVectorStore.from_documents(
        documents=all_docs,
        embedding=bedrock_embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    return vectorstore


def upload_chunks_from_s3(s3_key, bedrock_embeddings):
    # Download file from S3
    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    file_bytes = obj['Body'].read()

    # Save locally or load into pandas directly if Excel/CSV
    import io
    if s3_key.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    elif s3_key.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type for chunking")

    # If Excel with multiple sheets, you can adapt to split each sheet
    # For simplicity, let's say just one dataframe here for demo
    if isinstance(df, dict):  # multiple sheets
        dfs = []
        for sheet_name, sheet_df in df.items():
            dfs.append((sheet_name, sheet_df))
    else:
        dfs = [("Sheet1", df)]

    all_docs = []
    for sheet_name, sheet_df in dfs:
        #chunks = split_by_test_number(sheet_df, sheet_name, s3_key)
        chunks = split_by_tokens(df, sheet_name, uploaded_file.name)
        all_docs.extend(chunks)

    vectorstore = PineconeVectorStore.from_documents(
        documents=all_docs,
        embedding=bedrock_embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore


# CHUNKING LOGIC (tokens and test number)
def split_by_tokens(df, sheet_name: str, filename: str, max_tokens: int = 3000, overlap_tokens: int = 100) -> List[Document]:
    """Split DataFrame by token count, returning Document objects directly"""
    documents = []
    start_idx = 0
    
    while start_idx < len(df):
        current_chunk_rows = []
        current_token_count = 0
        end_idx = start_idx
        
        # Build chunk row by row until token limit is reached
        for idx in range(start_idx, len(df)):
            row_text = ' '.join([str(cell) for cell in df.iloc[idx].values if pd.notna(cell)])
            row_tokens = count_tokens(row_text)
            
            # Check if adding this row would exceed limit
            if current_token_count + row_tokens > max_tokens and current_chunk_rows:
                break
            
            current_chunk_rows.append(idx)
            current_token_count += row_tokens
            end_idx = idx + 1
        
        # If no rows fit (single row too large), take it anyway
        if not current_chunk_rows:
            current_chunk_rows = [start_idx]
            end_idx = start_idx + 1
        
        # Create chunk DataFrame
        chunk_df = df.iloc[current_chunk_rows]
        start_row = start_idx + 1  # 1-based row numbering
        end_row = end_idx
        
        # Convert DataFrame to text content
        content = f"Sheet: {sheet_name}\nSource: {filename}\nRows {start_row}-{end_row}:\n\n"
        content += chunk_df.to_string(index=True)
        
        # Calculate actual token count for the content
        actual_token_count = count_tokens(content)
        
        # Create Document object
        doc = Document(
            page_content=content,
            metadata={
                "source": filename,
                "sheet": sheet_name,
                "chunk_type": "Token Count",
                "start_row": start_row,
                "end_row": end_row,
                "total_rows": len(chunk_df),
                "token_count": actual_token_count,
                #"part_number": len(documents) + 1
            }
        )
        documents.append(doc)
        
        # Calculate overlap for next chunk
        if end_idx >= len(df):
            break
            
        # Find overlap starting position
        overlap_rows = min(overlap_tokens // 10, len(current_chunk_rows) // 4, 5)  # Rough estimate
        start_idx = max(start_idx + 1, end_idx - overlap_rows)
    
    # Add total_parts to all documents
    for doc in documents:
        doc.metadata["total_parts"] = len(documents)
    
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
                "chunk_type": "Full Sheet",
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
                "test_number": "Preamble",
                "chunk_type": "Test Number",
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
                "chunk_type": "Test Number",
                "start_row": start_row + 1,
                "end_row": end_row,
                "total_rows": len(chunk_df),
                "test_number": test_number
            }
        )
        chunks.append(chunk)
    
    return chunks

def extract_test_number(first_row):
    """Extract the test number from the first row"""
    for cell in first_row.values:
        if pd.notna(cell) and 'TestNumber' in str(cell):
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
    """Convert a test chunk to structured text
    Formats only non-empty cells into "ColumnName: Value" form.
    If a row has multiple non-empty columns, they’re joined with |.
    """
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

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken encoding"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except:
        # Fallback: rough approximation (4 chars per token)
        return len(text) // 5


# INITIAL PROTOTYPE (for local FAISS database)    
def get_vector_store(docs, embedding_model):
    # convert to embeddings
    vectorstore_faiss=FAISS.from_documents(
        docs,
        embedding_model # titan model
    )
    vectorstore_faiss.save_local("faiss_index")

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