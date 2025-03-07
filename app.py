import os
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from utils import ask_gpt_with_context, create_vector_store, extract_text_and_split

load_dotenv(find_dotenv("./env/.env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def main():
    st.title("PDF Quiz Generator")

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        # Only process PDF if vector store doesn't exist
        num_questions = st.slider(
            "Number of questions", min_value=5, max_value=20, value=5
        )
        if st.session_state.vector_store is None:
            with st.spinner("Processing PDF..."):
                progress_bar = st.progress(0)
                chunks = extract_text_and_split(
                    uploaded_file, progress_callback=progress_bar.progress
                )
                progress_bar.empty()
                st.success("PDF uploaded successfully!")

                # Store in session state
                st.session_state.vector_store = create_vector_store(chunks)

        # Generate Quiz button
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                quiz = ask_gpt_with_context(
                    uploaded_file.name,
                    st.session_state.vector_store,
                    client,
                    num_questions,
                )

                # Display the quiz with better formatting
                st.write("Generated Quiz:")
                # Split the quiz into individual questions

                for question in quiz.questions:
                    # Format each question
                    question_text = (
                        f"Difficulty: {question.difficulty}\n\n"
                        f"Question: {textwrap.fill(question.question, width=80)}\n\n"
                        f"Answer: {textwrap.fill(question.answer, width=80)}"
                    )

                    st.markdown(f"```\n{question_text}\n```")
                    st.markdown("---")

                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)

                # Format quiz for file output
                formatted_quiz = "\n\n".join(
                    f"Difficulty: {q.difficulty}\n"
                    f"Question: {q.question}\n"
                    f"Answer: {q.answer}"
                    for q in quiz.questions
                )

                quiz_file = (
                    output_dir / f"{uploaded_file.name.replace('.pdf', '')}_quiz.txt"
                )
                with open(quiz_file, "w", encoding="utf-8") as f:
                    f.write(formatted_quiz)

                st.success(f"Quiz saved to {quiz_file}")


if __name__ == "__main__":
    main()
