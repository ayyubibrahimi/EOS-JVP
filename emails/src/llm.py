import os
import logging
import pandas as pd
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import find_dotenv, load_dotenv
from datetime import datetime
import multiprocessing

load_dotenv(find_dotenv())

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["sender"] = record.get("sender")
    metadata["datetime"] = record.get("datetime")
    return metadata

def preprocess_document(file_path):
    logger.info(f"Processing document: {file_path}")
    loader = JSONLoader(file_path, jq_schema=".messages[]", content_key="page_content", metadata_func=metadata_func)
    documents = loader.load()
    return documents

template = """
<task_description>
As a Legal Clerk, your task is to review the email content and either summarize information related to the buying and selling of bonds, or indicate if the email is not relevant to this topic.
</task_description>

<guidelines>
1. Determine if the email contains any information related to buying and selling bonds.
2. If relevant, summarize the main points of the email as they pertain to bond transactions.
3. Include relevant details about the communication process, any mentioned parties, and general transaction information.
4. DO NOT include any details not explicitly stated in the email.
5. If the email does not contain relevant information, state this clearly.
</guidelines>

<essential_information>
Your objective is to either provide a clear, concise summary of bond-related content or explicitly state that the email does not contain relevant information.
</essential_information>

<thinking_process>
Before responding, consider:
1. Does this email contain any information about buying or selling bonds?
2. If so, what is the main purpose of this email regarding bond transactions?
3. If not, how can I clearly state its irrelevance to the topic?
</thinking_process>

<output_format>
If the email contains relevant information:
Provide a single, cohesive paragraph summarizing the email's content related to bond buying and selling. Focus on conveying the overall message and key points.

If the email does not contain relevant information:
Respond with only this sentence: "This email does not contain any correspondence related to the buying and selling of bonds."
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the email
</warnings>

<reference_materials>
## Email Content ##
{document}
</reference_materials>

<output_instruction>
Based on the email content, either provide a summary of bond buying and selling information or indicate its irrelevance below:
</output_instruction>
"""

def process_document(doc):
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()

    page_content = doc.page_content.replace('\n', ' ')
    sender = doc.metadata.get('sender')
    datetime_str = doc.metadata.get('datetime')
    if page_content:
        response = response_chain.invoke({"document": page_content})
        return {
            "original_email": page_content,
            "summary": response,
            "sender": sender,
            "datetime": datetime_str
        }
    return None

def get_response_from_query(docs):
    with multiprocessing.Pool() as pool:
        results = pool.map(process_document, docs)
    
    # Filter out None results
    return [result for result in results if result is not None]

def parse_datetime(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

def save_to_csv(results, output_file):
    df = pd.DataFrame(results)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by='datetime')
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    file_path = "../data/output/emails.json"
    output_file = "../data/output/email_summaries.csv"
    
    documents = preprocess_document(file_path)
    results = get_response_from_query(documents)
    save_to_csv(results, output_file)