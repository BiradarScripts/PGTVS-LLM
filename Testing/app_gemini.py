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

def main():
    # Specify the path to your .txt file
    file_path = "/workspaces/PGTVS-LLM/transcription.txt"

    try:
        # Read the content from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()

        if input_text:
            # Execute the LLM with the file content
            result = chain.invoke({"text": input_text})

            # Display the result
            print(f"AI Summary:\n{result}\n")

    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the file exists.")
    except ConnectionError:
        print("Failed to connect to the service. Please ensure the service is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
 