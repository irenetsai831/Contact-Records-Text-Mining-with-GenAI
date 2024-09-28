import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate, FewShotPromptTemplate

# Initialize the LLM
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_base=os.environ['CHATGPT_API_ENDPOINT'],
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0.1
)

# Create the First FewShotPromptTemplate
def create_few_shot_prompt_template():

    examples = [
        {"input": "家庭戶 10629 47915 145064", "output": "10629, 47915, 145064"},
        {"input": "家庭戶 王O強 12345 李O華", "output": "12345"},
        {"input": "家庭戶 張O明 王O芳 陳O偉", "output": "NA"},
    ]

    prefix = """
    請標註出文字中的的戶號。
    如果沒有戶號，請標註NA。
    """

    suffix = """
    Input: {input}
    Output:
    """

    example_template = """
    Input: {input}
    Output: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix=suffix,
        prefix=prefix,
        input_variables=["input"]
    )

    return few_shot_prompt_template

# Create the Second FewShotPromptTemplate
def create_second_few_shot_template():

    examples = [
        {"input": "家庭戶 10629 47915 145064", "output": "NA"},
        {"input": "家庭戶 王O強 12345 李O華", "output": "王O強, 李O華"},
        {"input": "家庭戶 張O明 王O芳 陳O偉", "output": "張O明, 王O芳, 陳O偉"},
    ]
  
    prefix = """
    請標註出文字中的的中文姓名。
    如果沒有中文姓名，請標註NA。
    """

    suffix = """
    Input: {input}
    Output:
    """

    example_template = """
    Input: {input}
    Output: {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    )

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix=suffix,
        prefix=prefix,
        input_variables=["input"]
    )

    return few_shot_prompt_template

# Function to classify the main contact with two steps
def classify_bfno_name(text):

    # reording which row they are
    row_process_total = 0
    row_process_unit = 1
    row_process_total += row_process_unit
    print(f"目前已經進行到第{row_process_total}行")
    
    # Create both few-shot templates
    first_template = create_few_shot_prompt_template()
    second_template = create_second_few_shot_template()
    
    # Use text as input, which extracts both "Customer_No" and "Name"
    frist_prompt = first_template.format(input=text)
    frist_output = chat.invoke(frist_prompt).content.strip() 

    second_prompt = second_template.format(input=text)
    second_output = chat.invoke(second_prompt).content.strip()  

    return {"Customer_No": frist_output, "Name": second_output}
