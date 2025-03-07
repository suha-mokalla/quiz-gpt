from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from PyPDF2 import PdfReader
from tqdm import tqdm


class Question(BaseModel):
    difficulty: str  # "Easy", "Medium", or "Hard"
    question: str
    answer: str


class Quiz(BaseModel):
    questions: list[Question]


def extract_text_and_split(pdf_path, progress_callback=None):
    reader = PdfReader(pdf_path)
    text = ""
    total_pages = len(reader.pages)

    if progress_callback:
        # Streamlit progress
        for i, page in enumerate(reader.pages):
            text += page.extract_text()
            progress_callback((i + 1) / total_pages)
    else:
        # CLI progress with tqdm
        for page in tqdm(reader.pages, desc="Extracting text from PDF"):
            text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def ask_gpt_with_context(query, vector_store, client, k=3, num_questions=5):
    relevant_docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": f"""Use the following context to create a quiz with {num_questions} questions.
                Make 40% of the questions easy, 40% medium, and 20% hard.
                Context:
                {context}
                """,
            }
        ],
        response_format=Quiz,
    )

    if completion.choices[0].message.refusal:
        return completion.choices[0].message.refusal

    return completion.choices[0].message.parsed
