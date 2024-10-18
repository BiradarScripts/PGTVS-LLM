import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

generic_template = '''I am providing you with a transcribed narration from a video, complete with
timestamps. Please generate an extractive summary from this text. Here are
your instructions:
1. The summary should consist of only the most critical and informative moments
from the video.
2. Do not paraphrase or reword the sentences. Maintain their original wording.
3. Each sentence you extract for the summary must include its original timestamp
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

def save_summary_to_file(output_file, content):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Summary has been written to {output_file}")
    except Exception as e:
        print(f"Failed to write summary to file: {e}")

def main():
    input_file = "transcript.txt"
    output_file = "gemini_result.txt"  

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            input_text = file.read()

        if input_text:
            result = chain.invoke({"text": input_text})
            save_summary_to_file(output_file, result)  

    except FileNotFoundError:
        print(f"File not found: {input_file}. Please ensure the file exists.")
    except ConnectionError:
        print("Failed to connect to the service. Please ensure the service is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
