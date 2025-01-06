import openai

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your actual API key


def ask_openai(prompt, max_tokens=300):
    """
    Interact with the OpenAI API using the latest chat-based models.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the latest model
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()


def gather_candidate_info():
    """
    Use OpenAI to gather initial candidate information.
    """
    prompt = """
    You are a hiring assistant chatbot for TalentScout. Start the conversation by greeting the candidate and introducing your purpose. 
    Then, ask the following questions one by one, ensuring a friendly and professional tone:
    - Full Name
    - Email Address
    - Phone Number
    - Years of Experience
    - Desired Position(s)
    - Current Location
    - Tech Stack (e.g., Python, React, SQL, etc.)
    Ensure to confirm each response before moving to the next question. If the candidate enters 'exit' or 'quit,' gracefully end the conversation.
    """
    return ask_openai(prompt)


def generate_technical_questions(tech_stack):
    """
    Generate technical questions based on the candidate's tech stack.
    """
    prompt = f"""
    You are a technical recruiter. A candidate has mentioned their proficiency in the following technologies: {tech_stack}.
    Generate 3-5 technical questions for each technology to assess their knowledge and skills.
    Ensure the questions are clear, relevant, and moderately challenging.
    """
    return ask_openai(prompt)


def handle_fallback(input_text):
    """
    Handle unexpected user inputs.
    """
    if input_text.lower() in ["exit", "quit"]:
        return "Thank you for chatting! Goodbye!"
    else:
        return "I'm sorry, I didn't understand that. Could you please clarify?"