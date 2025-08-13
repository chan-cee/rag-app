from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional
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
                        "temperature": kwargs.get("temperature", 0.2),
                        "top_p": kwargs.get("top_p", 0.5),
                        "max_completion_tokens": kwargs.get("max_completion_tokens", 4096)
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
                        "temperature": kwargs.get("temperature", 0.2),
                        "top_p": kwargs.get("top_p", 0.5),
                        "max_tokens": kwargs.get("max_tokens", 4096)
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