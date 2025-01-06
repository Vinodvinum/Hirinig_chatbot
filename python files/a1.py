import openai
import streamlit as st
import shelve

# OpenAI API key setup
openai.api_key = "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"

st.title("TalentScout Hiring Assistant Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Ensure chat history is initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history()

# Sidebar to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state["messages"] = []
        save_chat_history([])

# --- Step 1: Collect Candidate Information ---
if "info_collected" not in st.session_state:
    st.session_state["info_collected"] = False
    st.session_state["candidate_info"] = {}

# Candidate information form
if not st.session_state["info_collected"]:
    with st.form("Candidate Info Form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        experience = st.number_input("Years of Experience", min_value=0, step=1)
        tech_stack = st.text_area("Tech Stack (e.g., Python, Django, MySQL)").split(", ")
        submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state["candidate_info"] = {
                "Name": name,
                "Email": email,
                "Phone": phone,
                "Experience": experience,
                "Tech Stack": tech_stack,
            }
            st.session_state["info_collected"] = True
            st.success("Candidate information collected!")

# --- Step 2: Generate Technical Questions ---
if st.session_state["info_collected"]:
    st.write("### Candidate Information:")
    for key, value in st.session_state["candidate_info"].items():
        st.write(f"**{key}:** {value}")

    tech_stack = st.session_state["candidate_info"]["Tech Stack"]
    if "questions" not in st.session_state:
        st.session_state["questions"] = {}

    # Generate questions dynamically for each technology
    for tech in tech_stack:
        if tech not in st.session_state["questions"]:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert interviewer."},
                    {"role": "user", "content": f"Generate 3 interview questions for {tech}."},
                ],
            )
            st.session_state["questions"][tech] = response["choices"][0]["message"]["content"]

    # Display generated questions
    st.write("### Technical Questions:")
    for tech, q in st.session_state["questions"].items():
        st.markdown(f"#### {tech}")
        st.markdown(q)

# --- Step 3: Chatbot Interaction ---
st.write("### Chat with TalentScout Assistant")

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("Ask me anything about the hiring process!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Chatbot response logic
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)