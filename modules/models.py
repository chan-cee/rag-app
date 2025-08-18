from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional
import json
from pydantic import Field
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
#from langchain_core.documents import Document
#from langchain_core.rerankers import BaseDocumentCompressor
import boto3, json
import uuid

#custom wrapper
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
                        "temperature": kwargs.get("temperature", 0.2),
                        "top_p": kwargs.get("top_p", 0.5),
                        "max_completion_tokens": kwargs.get("max_completion_tokens", 8192)
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

# class GPTLLM(BaseLLM):
#     bedrock: Any = Field(exclude=True)
#     model_id: str = "openai.gpt-oss-120b-1:0"
    
#     def _generate(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> LLMResult:
        
#         # DEBUG: Add logging to see what's happening
#         request_id = str(uuid.uuid4())[:8]
#         print(f"🔍 GPTLLM._generate called with {len(prompts)} prompts (ID: {request_id})")
        
#         # Log each prompt (first 100 chars)
#         for i, prompt in enumerate(prompts):
#             print(f"  Prompt {i+1}: {prompt[:100]}...")
        
#         generations = []
        
#         for i, prompt in enumerate(prompts):
#             try:
#                 print(f"📡 Making Bedrock API call {i+1}/{len(prompts)} (ID: {request_id})")
                
#                 response = self.bedrock.invoke_model(
#                     modelId=self.model_id,
#                     contentType="application/json",
#                     accept="application/json",
#                     body=json.dumps({
#                         "messages": [{"role": "user", "content": prompt}],
#                         "temperature": kwargs.get("temperature", 0.2),
#                         "top_p": kwargs.get("top_p", 0.5),
#                         "max_completion_tokens": kwargs.get("max_completion_tokens", 4000)  # Reduced from 8192
#                     })
#                 )
                
#                 response_body = json.loads(response['body'].read())
#                 text = response_body['choices'][0]['message']['content']
#                 generations.append([Generation(text=text)])
                
#                 print(f"✅ API call {i+1} completed (ID: {request_id})")
                
#             except Exception as e:
#                 print(f"❌ API call {i+1} failed: {str(e)} (ID: {request_id})")
#                 generations.append([Generation(text=f"Error: {str(e)}")])
        
#         print(f"🏁 GPTLLM._generate completed for ID: {request_id}")
#         return LLMResult(generations=generations)
    
#     @property
#     def _llm_type(self) -> str:
#         return "gpt-oss"

    
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
                        "temperature": kwargs.get("temperature", 0.2),
                        "top_p": kwargs.get("top_p", 0.5),
                        "max_tokens": kwargs.get("max_tokens", 8192) #4096
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



