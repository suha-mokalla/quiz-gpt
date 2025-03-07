import os
import random
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from utils import (
    ask_gpt_with_context,
    check_answer,
    create_vector_store,
    extract_text_and_split,
    generate_single_question,
)

load_dotenv(find_dotenv("./env/.env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def main():
    st.title("PDF Quiz Generator")

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_explanations" not in st.session_state:
        st.session_state.current_explanations = None

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        # Only process PDF if vector store doesn't exist
        tab1, tab2 = st.tabs(["Interactive Quiz", "Batch Quiz"])

        with tab1:
            # Only process PDF if vector store doesn't exist
            if st.session_state.questions_asked > 0:
                st.info(
                    f"Current Score: {st.session_state.score}/{st.session_state.questions_asked}"
                )
            if st.session_state.vector_store is None:
                with st.spinner("Processing PDF..."):
                    progress_bar = st.progress(0)
                    chunks = extract_text_and_split(
                        uploaded_file, progress_callback=progress_bar.progress
                    )
                    progress_bar.empty()
                    st.success("PDF uploaded successfully!")
                    st.session_state.vector_store = create_vector_store(chunks)

            # Difficulty selector
            difficulty = st.selectbox(
                "Select difficulty level",
                options=["Easy", "Medium", "Hard"],
                key="difficulty",
            )

            # Generate Question button
            if st.button("Generate Question"):
                st.session_state.answer_submitted = False
                with st.spinner("Generating question..."):
                    try:
                        random_query = random.choice(
                            ["what", "how", "why", "describe", "explain"]
                        )
                        relevant_docs = st.session_state.vector_store.similarity_search(
                            random_query, k=2
                        )
                        context = "\n".join([doc.page_content for doc in relevant_docs])
                        question = generate_single_question(
                            context,
                            difficulty,
                            client,
                        )
                        st.session_state.current_question = question
                    except ValueError as e:
                        st.error(f"Error generating question: {e}")

            # Display current question if it exists
            if st.session_state.current_question:
                st.write("### Question:")
                st.write(st.session_state.current_question.question)

                # Radio buttons for options
                selected_option = st.radio(
                    "Choose your answer:",
                    options=range(len(st.session_state.current_question.options)),
                    format_func=lambda x: st.session_state.current_question.options[x],
                    key="answer",
                    index=None,
                )

                if "answer_submitted" not in st.session_state:
                    st.session_state.answer_submitted = False

            # Submit button
            if st.button("Submit Answer"):
                is_correct, feedback = check_answer(
                    st.session_state.current_question, selected_option
                )

                if is_correct:
                    st.session_state.score += 1
                st.session_state.questions_asked += 1

                # Display feedback
                st.markdown(feedback)
                st.markdown(
                    f"**Current Score:** {st.session_state.score}/{st.session_state.questions_asked}"
                )

        with tab2:
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
                        output_dir
                        / f"{uploaded_file.name.replace('.pdf', '')}_quiz.txt"
                    )
                    with open(quiz_file, "w", encoding="utf-8") as f:
                        f.write(formatted_quiz)

                    st.success(f"Quiz saved to {quiz_file}")


if __name__ == "__main__":
    main()
