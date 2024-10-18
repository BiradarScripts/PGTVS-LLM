import os
from dotenv import load_dotenv
import requests
from requests.exceptions import ConnectionError

from langchain_ollama import OllamaLLM  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

generic_template = '''
I am providing you with a transcribed narration from a video, complete with
timestamps. Please generate an extractive summary from this text. Here are
your instructions:
1. The summary should consist of only the most critical and informative moments
from the video.
2. Do not paraphrase or reword the sentences. Maintain their original wording.
3. Each sentence you extract for the summary must include its original timestamp.
'''

def load_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def save_summary_to_file(output_file, content):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Summary has been written to {output_file}")
    except Exception as e:
        print(f"Failed to write summary to file: {e}")

input_file = 'transcript.txt'  
output_file = 'ollama_result.txt' 

transcribed_text = load_file_content(input_file)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", f"{generic_template}\nTranscription:\n{transcribed_text}")
    ]
)

llm = OllamaLLM(model="llama3.1")  
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

try:
    result = chain.invoke({})
    save_summary_to_file(output_file, result)  
except ConnectionError:
    print("Failed to connect to the Ollama service. Please ensure the service is running.")
except Exception as e:
    print(f"An error occurred: {e}")
