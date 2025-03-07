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


class SingleQuestion(BaseModel):
    question: str
    correct_answer: str
    options: list[str]
    explanations: list[str]


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


def generate_single_question(context: str, difficulty: str, client) -> SingleQuestion:
    """
    Generate a single multiple-choice question with options based on the given context and difficulty.

    Args:
        context: The text content to base the question on
        difficulty: "Easy", "Medium", or "Hard"
        client: OpenAI client instance

    Returns:
        SingleQuestion object containing the question, correct answer, and multiple choice options
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """Generate a multiple-choice question based on the provided context. 
                For Easy questions: focus on basic facts and definitions.
                For Medium questions: test understanding and relationships between concepts.
                For Hard questions: require analysis, synthesis, or application of multiple concepts
                For each option, provide a clear explanation of why it is correct or incorrect.""",
            },
            {
                "role": "user",
                "content": f"""Difficulty level: {difficulty}
                Context: {context}
                Create one multiple-choice question. Make sure:
                1. The question is clear and specific
                2. There are exactly 4 options
                3. Only one option is correct
                4. All options are plausible
                5. Options are shuffled (correct answer not always first)
                6. Include a brief explanation for why each option is correct or incorrect""",
            },
        ],
        response_format=SingleQuestion,
    )

    if completion.choices[0].message.refusal:
        raise ValueError(completion.choices[0].message.refusal)

    return completion.choices[0].message.parsed


def check_answer(question: SingleQuestion, answer_idx: int) -> tuple[bool, str]:
    selected_option = question.options[answer_idx]
    is_correct = selected_option == question.correct_answer

    feedback = []
    if is_correct:
        feedback.append("✅ Correct!")
    else:
        feedback.append(
            f"❌ Incorrect. The correct answer was: {question.correct_answer}"
        )

    feedback.append("\nHere's why:")
    for option, explanation in zip(question.options, question.explanations):
        feedback.append(f"\n{option}: {explanation}")

    return is_correct, "\n".join(feedback)
