from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from .prompts import ICD_SEMANTIC_PROMPT
from .schema import ICDLLMResponse
from utils.config import GOOGLE_API_KEY
import time

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    request_timeout=60,   # prevent hanging
)

parser = PydanticOutputParser(pydantic_object=ICDLLMResponse)

def extract_icd_from_chunk(chunk: str, max_retries: int = 2):
    """
    Extract ICD codes and their conditions from chunk using LLM with retry logic.
    Returns tuple: (list of ICD codes, list of Diagnosis objects with condition+code)
    """
    # Safety: Skip very small chunks
    if len(chunk.strip()) < 30:
        return [], []

    for attempt in range(max_retries):
        try:
            prompt = ICD_SEMANTIC_PROMPT.format(chunk=chunk)
            response = llm.invoke(prompt)

            parsed = parser.parse(response.content)
            icd_codes = [d.icd10 for d in parsed.diagnoses]
            diagnoses = parsed.diagnoses  # Keep full objects
            return icd_codes, diagnoses

        except Exception as e:
            time.sleep(2)  # small backoff
            if attempt == max_retries - 1:
                return [], []

    return [], []
