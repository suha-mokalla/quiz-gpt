import streamlit as st
import textwrap
import os
from pathlib import Path
from utils import extract_text_and_split, create_vector_store, ask_gpt_with_context
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv("./env/.env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def main():
    st.title("PDF Quiz Generator")
    
    # Initialize session state for vector store
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file:
        # Only process PDF if vector store doesn't exist
        num_questions = st.slider("Number of questions", min_value=5, max_value=20, value=5)
        if st.session_state.vector_store is None:
            with st.spinner('Processing PDF...'):
                progress_bar = st.progress(0)
                chunks = extract_text_and_split(uploaded_file, progress_callback=progress_bar.progress)
                progress_bar.empty()
                st.success("PDF uploaded successfully!")
                
                # Store in session state
                st.session_state.vector_store = create_vector_store(chunks)
        
        # Generate Quiz button
        if st.button("Generate Quiz"):
            with st.spinner('Generating quiz...'):
                quiz = ask_gpt_with_context(uploaded_file.name, st.session_state.vector_store, client, num_questions)
                
                # Display the quiz with better formatting
                st.write("Generated Quiz:")
                # Split the quiz into individual questions
                questions = quiz.strip().split("\n\n")
                
                for question in questions:
                    # Split into components
                    lines = question.split("\n")
                    formatted_lines = []
                    
                    for line in lines:
                        # If line starts with "Question:" or "Answer:", wrap the text
                        if line.startswith(("Question:", "Answer:")):
                            label, content = line.split(":", 1)
                            # Wrap text to 80 characters, with indentation for wrapped lines
                            wrapped_content = "\n    ".join(textwrap.fill(content.strip(), width=80).split("\n"))
                            formatted_lines.append(f"{label}: {wrapped_content}")
                        else:
                            formatted_lines.append(line)
                    
                    formatted_question = "\n\n".join(formatted_lines)
                    st.markdown(f"```\n{formatted_question}\n```")
                    st.markdown("---")  # Add a separator line
                
                # Save the quiz to file
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                
                quiz_file = output_dir / f"{uploaded_file.name.replace('.pdf', '')}_quiz.txt"
                with open(quiz_file, "w", encoding="utf-8") as f:
                    f.write(quiz)
                
                st.success(f"Quiz saved to {quiz_file}")


if __name__ == "__main__":
    main()