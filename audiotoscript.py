import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import pathlib

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

def main():

    media_path = pathlib.Path("sample2.wav")  

    try:

        with open(media_path, 'rb') as audio_file:
            audio_data = audio_file.read()


        transcription_file_path = media_path.with_suffix('.txt')  

        with open(transcription_file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()

        if input_text:

            prompt = generic_template + input_text


            result = llm.invoke([
                {
                    "mime_type": "audio/wav", 
                    "data": audio_data
                },
                prompt
            ])


            output_file_path = media_path.with_name("summary.txt") 
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(result.text)
            print(f"Summary saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"File not found: {transcription_file_path}. Please ensure the file exists.")
    except ConnectionError:
        print("Failed to connect to the service. Please ensure the service is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
