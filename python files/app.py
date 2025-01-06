import streamlit as st
from openai_integration import ask_openai, gather_candidate_info, generate_technical_questions

# App title
st.title("TalentScout Hiring Assistant")

# App description
st.write("Welcome to the TalentScout Hiring Assistant! Let's gather candidate information and generate some technical questions.")

# Candidate information section
if st.button("Start Candidate Information Gathering"):
    st.write("Gathering Candidate Information...")
    candidate_info = gather_candidate_info()
    st.text(candidate_info)

# Generate technical questions
tech_stack = st.text_input("Enter the technologies (comma-separated, e.g., Python, Django, SQL):")
if st.button("Generate Technical Questions"):
    if tech_stack:
        st.write("Generating Technical Questions...")
        questions = generate_technical_questions(tech_stack.split(","))
        st.text(questions)
    else:
        st.warning("Please enter at least one technology to generate questions.")

# End the chat
if st.button("End Chat"):
    st.write("Thank you for using the TalentScout Hiring Assistant! Goodbye!")