import argparse
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from utils import ask_gpt_with_context, create_vector_store, extract_text_and_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main(args):
    load_dotenv(find_dotenv("./env/.env"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chunks = extract_text_and_split(args.pdf_path)
    vector_store = create_vector_store(chunks)
    doc_name = Path(args.pdf_path).name
    logging.info(f"PDF loaded! Generating quiz for {doc_name}")
    quiz = ask_gpt_with_context(args.pdf_path, vector_store, client, args.num_questions)
    logging.info("Quiz generated!")
    formatted_quiz = "\n\n".join(
        f"Difficulty: {q.difficulty}\nQuestion: {q.question}\nAnswer: {q.answer}"
        for q in quiz.questions
    )
    logging.info(formatted_quiz)

    # Create outputs directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Create quiz file with formatted content
    quiz_file = output_dir / f"{doc_name.replace('.pdf', '')}_quiz.txt"
    with open(quiz_file, "w", encoding="utf-8") as f:
        f.write(formatted_quiz)

    logging.info(f"\nQuiz saved to {quiz_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions about a PDF document")
    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the PDF file"
    )
    parser.add_argument(
        "--num_chunks", type=int, default=3, help="Number of chunks to use for context"
    )
    parser.add_argument(
        "--num_questions", type=int, default=5, help="Number of questions to create"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
