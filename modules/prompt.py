from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

system_template = (
    "You are an expert in semiconductor test automation and script translation. "
    "Your task is to convert legacy MCT 2000 test scripts into the modern SPEAL format. "
    "You must maintain the logic, structure, and intent of the original script while rewriting it using SPEAL syntax."
    "\nUse the following context as reference if helpful. Code on the LEFT represents SPEAL formatting, while code on the RIGHT represents MCT 2000 formatting.\n"
    "Always output the converted code in the code editor with exact syntax and formatting for easy viewing or copying."
)

human_template = """
<context>
{context}
</context>

Question: {question}
"""

def get_prompt():
    PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return PROMPT