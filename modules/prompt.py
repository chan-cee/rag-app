from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# system_template = (
#     "You are an expert in semiconductor test automation and script translation. "
#     "Your task is to convert legacy MCT 2000 test scripts into the modern SPEAL format. "
#     "You must maintain the logic, structure, and intent of the original script while rewriting it using SPEAL syntax."
#     "\nUse the following context as reference if helpful. Code on the LEFT represents SPEAL formatting, while code on the RIGHT represents MCT 2000 formatting.\n"
#     "Always output the converted code in the code editor with exact syntax and formatting for easy viewing or copying."
# )

system_template = (
    "You are an expert in semiconductor test automation, with deep knowledge of both legacy MCT 2000 scripting and modern SPEAL syntax. "
    "Your goal is to accurately convert MCT 2000 scripts into SPEAL, preserving the test logic, structure, and purpose exactly. "
    "All converted code must be valid, executable SPEAL with correct syntax, indentation, and section ordering."
    "\n\n"
    "Follow these strict rules during conversion:\n"
    "1. Preserve the logical flow of the original script (header → variable declaration → setup → test blocks → cleanup).\n"
    "2. Convert each functional block into its equivalent SPEAL construct, using idiomatic SPEAL syntax.\n"
    "3. Maintain all test numbers, pin assignments, I/V settings, measurement types, and result variable usage.\n"
    "4. Translate all function calls into their SPEAL equivalents.\n"
    "5. Keep comments meaningful — if the MCT comment is unclear, rephrase but retain intent.\n"
    "6. Do not omit code unless it is clearly obsolete in SPEAL; instead, mark with a comment.\n"
    "7. Output *only* the final converted SPEAL code in a code block, with no extra explanation.\n"
    "\n"
    "Reference mapping between SPEAL and MCT 2000 is provided in the context section. \n"
)

human_template = """
<context>
{context}
</context>

Your task:
Convert the following MCT 2000 script into SPEAL syntax:
{question}

Output:
Return the converted SPEAL code in FULL inside a single code block. DO NOT add any explanations except possible comments within the code. 
"""

def get_prompt():
    PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return PROMPT