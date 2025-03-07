import argparse
import logging
import os
import random
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from utils import (
    ask_gpt_with_context,
    check_answer,
    create_vector_store,
    extract_text_and_split,
    generate_single_question,
)

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

    if args.mode == "interactive":
        difficulty = input("Choose difficulty (Easy/Medium/Hard): ").capitalize()
        score = 0
        questions_asked = 0

        while True:
            # Get random context
            random_query = random.choice(["what", "how", "why", "describe", "explain"])
            relevant_docs = vector_store.similarity_search(random_query, k=2)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Generate question
            try:
                question = generate_single_question(context, difficulty, client)
            except ValueError as e:
                print(f"Error generating question: {e}")
                continue

            # Display question
            print(f"\nQuestion {questions_asked + 1}:")
            print(question.question)
            for i, option in enumerate(question.options, 1):
                print(f"{i}. {option}")

            # Get answer
            while True:
                answer = input("\nEnter your answer (1-4) or 'q' to quit: ").lower()
                if answer == "q":
                    break
                try:
                    answer_idx = int(answer) - 1
                    if 0 <= answer_idx < len(question.options):
                        break
                    print("Please enter a number between 1 and 4")
                except ValueError:
                    print("Please enter a valid number or 'q' to quit")

            if answer == "q":
                break
            questions_asked += 1
            is_correct, feedback = check_answer(question, answer_idx)
            if is_correct:
                score += 1

            print(feedback)
            print(f"\nCurrent score: {score}/{questions_asked}")

            # Ask to continue
            if input("\nPress Enter for next question or 'q' to quit: ").lower() == "q":
                break

        print(f"\nFinal Score: {score}/{questions_asked}")

    else:
        logging.info(f"PDF loaded! Generating quiz for {doc_name}")
        quiz = ask_gpt_with_context(
            args.pdf_path, vector_store, client, args.num_questions
        )
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
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "batch"],
        help="Mode of operation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
