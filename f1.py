import openai
import streamlit as st
import shelve
import re  # For regex validation
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI API key setup

openai.api_key = OPENAI_API_KEY
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

# Validate email format
def is_valid_email(email):
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(email_regex, email))

# Validate phone number format
def is_valid_phone(phone):
    phone_regex = r'^\+?\d{10,15}$'  # Accepts international and local formats
    return bool(re.match(phone_regex, phone))

# Modularized form validation
def validate_form(name, email, phone, experience, position, location, tech_stack):
    missing_fields = []
    if not name:
        missing_fields.append("Full Name")
    if not email or not is_valid_email(email):
        missing_fields.append("Valid Email Address")
    if not phone or not is_valid_phone(phone):
        missing_fields.append("Valid Phone Number")
    if experience is None or experience < 0:
        missing_fields.append("Years of Experience")
    if not position:
        missing_fields.append("Desired Position(s)")
    if not location:
        missing_fields.append("Current Location")
    if not tech_stack or "" in tech_stack:
        missing_fields.append("Tech Stack")
    return missing_fields

# Initialize chat history and other states
if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history()

if "info_collected" not in st.session_state:
    st.session_state["info_collected"] = False
    st.session_state["candidate_info"] = {}

# Sidebar to manage chat history
with st.sidebar:
    if st.button("Start New Chat"):
        st.session_state["messages"] = []
        st.session_state["info_collected"] = False
        st.session_state["candidate_info"] = {}
        st.session_state["greeted"] = False  # Reset the greeted
        save_chat_history([])
    if st.button("Delete Chat History"):
        st.session_state["messages"] = []
        save_chat_history([])

# --- Greeting ---
if not st.session_state.get("greeted"):
    st.session_state["greeted"] = True
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hi! Welcome to TalentScout. I’m your Hiring Assistant. Let’s get started! "
            "I can assist with collecting your details, generating tailored technical questions, "
            "and addressing your queries."
        )
    })
    save_chat_history(st.session_state.messages)

# Display chat history
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Collect Candidate Information ---
if not st.session_state["info_collected"]:
    with st.form("Candidate Info Form"):
        # Create form inputs with placeholders
        name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email Address", placeholder="Enter a valid email address")
        phone = st.text_input("Phone Number", placeholder="Enter your phone number (+91... or local)")
        experience = st.number_input("Years of Experience", min_value=0, step=1)
        position = st.text_input("Desired Position(s)", placeholder="e.g., Software Engineer, Data Scientist")
        location = st.text_input("Current Location", placeholder="Enter your current city")
        tech_stack = st.text_area(
            "Tech Stack (e.g., Python, Django, MySQL)",
            placeholder="Enter comma-separated technologies (e.g., Python, React, MySQL)"
        ).split(", ")

        # Check if form is submitted
        submitted = st.form_submit_button("Submit")

        # Validate form inputs
        if submitted:
            missing_fields = validate_form(name, email, phone, experience, position, location, tech_stack)

            # Show warnings if any field is invalid
            if missing_fields:
                for field in missing_fields:
                    st.warning(f"Please fill in your {field}.")
            else:
                # If all fields are valid, save the info and show confirmation
                st.session_state["candidate_info"] = {
                    "Name": name,
                    "Email": email,
                    "Phone": phone,
                    "Experience": experience,
                    "Position": position,
                    "Location": location,
                    "Tech Stack": tech_stack,
                }
                st.session_state["info_collected"] = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Thank you, {name}! Your information has been recorded. I’ll now generate technical questions based on your tech stack."
                })
                save_chat_history(st.session_state.messages)

# --- Display Candidate Information ---
if st.session_state["info_collected"]:
    st.write("### Candidate Information:")
    for key, value in st.session_state["candidate_info"].items():
        st.write(f"**{key}:** {value}")


# Predefined Question Bank for Technologies, Soft Skills, and Behavioral Questions (case-insensitive keys)
QUESTION_BANK = {
    # Programming Languages
    "python": [
        "What are Python's key features?",
        "Explain the difference between deep copy and shallow copy in Python.",
        "How does Python manage memory?",
        "What is the purpose of Python's `__init__` method?",
        "Can you explain Python's GIL (Global Interpreter Lock)?"
    ],
    "tensorflow": [
        "What is TensorFlow, and why is it widely used in AI/ML?",
        "Explain the concept of computational graphs in TensorFlow.",
        "What are TensorFlow tensors, and how do they differ from NumPy arrays?",
        "How do you optimize a model in TensorFlow using Keras?",
        "What is TensorFlow Serving, and how is it used?"
    ],
    "pytorch": [
        "What is PyTorch, and how does it differ from TensorFlow?",
        "Explain the concept of dynamic computational graphs in PyTorch.",
        "How do you perform backpropagation in PyTorch?",
        "What are PyTorch's DataLoader and Dataset classes used for?",
        "How do you handle GPU acceleration in PyTorch?"
    ],
    "opencv": [
        "What is OpenCV, and what are its key applications in computer vision?",
        "How do you perform edge detection using OpenCV?",
        "What are the differences between morphological operations like dilation and erosion in OpenCV?",
        "How do you handle object detection using OpenCV?",
        "What are contours in OpenCV, and how do you use them?"
    ],
    "spacy": [
        "What is SpaCy, and how does it differ from NLTK?",
        "Explain the pipeline components of SpaCy.",
        "How do you use SpaCy for Named Entity Recognition (NER)?",
        "What is a `Doc` object in SpaCy, and how do you process text with it?",
        "How do you train custom models in SpaCy?"
    ],
    "transformers": [
        "What is the Transformers library by Hugging Face?",
        "Explain the architecture of BERT and its use cases.",
        "How do you fine-tune a pretrained transformer model?",
        "What is the difference between GPT and BERT in terms of architecture?",
        "How do you use Transformers for text classification tasks?"
    ],

    "java": [
        "What are the key features of Java?",
        "Explain the concept of Java's 'platform independence.'",
        "What is the difference between an abstract class and an interface in Java?",
        "How does garbage collection work in Java?",
        "What are Java's access modifiers, and how are they used?"
    ],
    "c++": [
        "What are the main features of C++?",
        "Explain the concept of constructors and destructors in C++.",
        "What is the difference between 'new' and 'malloc' in C++?",
        "Explain the concept of polymorphism in C++ with examples.",
        "What is the purpose of virtual functions in C++?"
    ],
    "javascript": [
        "What is the difference between `var`, `let`, and `const` in JavaScript?",
        "Explain the concept of closures in JavaScript.",
        "What is the event loop in JavaScript?",
        "How does asynchronous programming work in JavaScript?",
        "What are promises, and how do they differ from async/await?"
    ],
    "html": [
        "What is the difference between HTML and XHTML?",
        "Explain the concept of semantic HTML.",
        "What are HTML5 web storage types, and how do they differ?",
        "What is the difference between `<div>` and `<section>` in HTML?",
        "What are meta tags, and why are they important?"
    ],
    "css": [
        "What is the difference between inline, internal, and external CSS?",
        "What are CSS selectors, and how are they used?",
        "Explain the concept of the CSS box model.",
        "What is the difference between relative, absolute, and fixed positioning in CSS?",
        "How do media queries work in responsive design?"
    ],

    # Frameworks and Libraries
    "django": [
        "What is Django, and why is it used?",
        "Explain the Django MVT architecture.",
        "How do you handle static files in Django?",
        "What are Django middleware, and how do they work?",
        "How do you perform database migrations in Django?"
    ],
    "react": [
        "What is JSX in React?",
        "Explain the concept of state and props in React.",
        "What is a React component lifecycle?",
        "How does React handle virtual DOM?",
        "What are React hooks, and why are they used?"
    ],
    "angular": [
        "What is Angular, and how is it different from React?",
        "Explain Angular's dependency injection system.",
        "What are Angular directives, and how are they used?",
        "How does Angular handle two-way data binding?",
        "What is the purpose of Angular modules?"
    ],
    "flask": [
        "What is Flask, and how is it different from Django?",
        "How do you handle sessions in Flask?",
        "What are Flask blueprints, and why are they used?",
        "How do you implement authentication in Flask?",
        "What are Flask extensions, and how do they work?"
    ],

    # Databases
    "mysql": [
        "What are the differences between MySQL and NoSQL databases?",
        "What is normalization, and why is it important?",
        "How do you optimize queries in MySQL?",
        "What is a foreign key, and how is it used?",
        "What are MySQL indexes, and how do they improve performance?"
    ],
    "mongodb": [
        "What is MongoDB, and how is it different from relational databases?",
        "Explain the concept of collections and documents in MongoDB.",
        "How does sharding work in MongoDB?",
        "What is the Aggregation Framework in MongoDB?",
        "How do you handle indexing in MongoDB?"
    ],

    # Cloud Technologies
    "aws": [
        "What is AWS, and why is it popular?",
        "What are the differences between EC2, S3, and Lambda?",
        "Explain the concept of load balancing in AWS.",
        "What is the purpose of AWS CloudFormation?",
        "How do you secure data in AWS S3 buckets?"
    ],
    "azure": [
        "What is Azure, and how does it differ from AWS?",
        "What is the purpose of Azure Resource Manager?",
        "Explain Azure's App Service and its uses.",
        "How does Azure handle authentication and identity management?",
        "What is Azure DevOps, and why is it used?"
    ],

    # Soft Skills
    "communication": [
        "Can you describe a time when you had to explain a technical concept to a non-technical person?",
        "How do you ensure clarity and transparency in communication within a team?",
        "What strategies do you use to handle misunderstandings in communication?",
        "How do you adapt your communication style for different audiences?",
        "Can you provide an example of a situation where effective communication led to a successful outcome?"
    ],
    "teamwork": [
        "Can you describe a time when you worked successfully as part of a team?",
        "How do you handle conflicts within a team?",
        "What steps do you take to ensure collaboration in a diverse team?",
        "How do you handle situations where a team member is not contributing effectively?",
        "What role do you typically take when working in a team, and why?"
    ],
    "leadership": [
        "Can you provide an example of a time when you led a team to success?",
        "How do you motivate team members to achieve their best performance?",
        "What is your approach to decision-making in a leadership role?",
        "How do you handle situations where you need to give constructive feedback?",
        "What strategies do you use to resolve conflicts as a leader?"
    ],

    # Behavioral Questions
    "behavioral": [
        "Can you describe a time when you faced a major challenge and how you overcame it?",
        "What steps do you take to prioritize tasks when handling multiple deadlines?",
        "Can you provide an example of a time when you received constructive criticism and how you acted on it?",
        "How do you handle stress in high-pressure situations?",
        "What motivates you to consistently perform at a high level?"
    ],

    "tensorflow": [
        "What is TensorFlow, and why is it widely used in AI/ML?",
        "Explain the concept of computational graphs in TensorFlow.",
        "What are TensorFlow tensors, and how do they differ from NumPy arrays?",
        "How do you optimize a model in TensorFlow using Keras?",
        "What is TensorFlow Serving, and how is it used?"
    ],
    "pytorch": [
        "What is PyTorch, and how does it differ from TensorFlow?",
        "Explain the concept of dynamic computational graphs in PyTorch.",
        "How do you perform backpropagation in PyTorch?",
        "What are PyTorch's DataLoader and Dataset classes used for?",
        "How do you handle GPU acceleration in PyTorch?"
    ],
    "opencv": [
        "What is OpenCV, and what are its key applications in computer vision?",
        "How do you perform edge detection using OpenCV?",
        "What are the differences between morphological operations like dilation and erosion in OpenCV?",
        "How do you handle object detection using OpenCV?",
        "What are contours in OpenCV, and how do you use them?"
    ],
    "spacy": [
        "What is SpaCy, and how does it differ from NLTK?",
        "Explain the pipeline components of SpaCy.",
        "How do you use SpaCy for Named Entity Recognition (NER)?",
        "What is a `Doc` object in SpaCy, and how do you process text with it?",
        "How do you train custom models in SpaCy?"
    ],
    "transformers": [
        "What is the Transformers library by Hugging Face?",
        "Explain the architecture of BERT and its use cases.",
        "How do you fine-tune a pretrained transformer model?",
        "What is the difference between GPT and BERT in terms of architecture?",
        "How do you use Transformers for text classification tasks?"
    ],

    "data analytics": [
        "What is the difference between descriptive and inferential statistics?",
        "How do you handle missing data in a dataset?",
        "What are the key differences between supervised and unsupervised learning?",
        "How do you use Python libraries like Pandas for data cleaning?",
        "What is the purpose of data visualization, and how do tools like Matplotlib and Seaborn help?"
    ],

    "data engineering": [
        "What is the role of a data engineer in an organization?",
        "Explain ETL (Extract, Transform, Load) processes in data pipelines.",
        "How do you ensure data quality in large-scale data processing?",
        "What are the differences between batch and stream processing?",
        "How do you optimize database queries for performance?"
    ],

    "data science": [
        "What is the CRISP-DM process, and how is it used in data science projects?",
        "How do you choose an appropriate machine learning algorithm for a problem?",
        "What are common methods for feature selection in machine learning?",
        "Explain the difference between precision and recall in classification problems.",
        "How do you handle imbalanced datasets in machine learning?"
    ],

    "ai/ml": [
        "What is the difference between AI, ML, and DL (Deep Learning)?",
        "How do you evaluate the performance of a machine learning model?",
        "What is hyperparameter tuning, and how is it done?",
        "Explain the concept of transfer learning in deep learning.",
        "How do you handle overfitting in machine learning models?"
    ],

    "ai": [
        "What is the difference between AI, ML, and DL (Deep Learning)?",
        "How do you evaluate the performance of a machine learning model?",
        "What is hyperparameter tuning, and how is it done?",
        "Explain the concept of transfer learning in deep learning.",
        "How do you handle overfitting in machine learning models?"
    ],

    "ml": [
        "What is the difference between AI, ML, and DL (Deep Learning)?",
        "How do you evaluate the performance of a machine learning model?",
        "What is hyperparameter tuning, and how is it done?",
        "Explain the concept of transfer learning in deep learning.",
        "How do you handle overfitting in machine learning models?"
    ],

    "full stack developer": [
        "What is the difference between monolithic and microservices architecture?",
        "How do you manage state in a full-stack application?",
        "What are REST APIs, and how are they used in full-stack development?",
        "What is the purpose of version control systems like Git in full-stack development?",
        "How do you ensure cross-browser compatibility in your applications?"
    ],

    "front end developer": [
        "What is the difference between client-side and server-side rendering?",
        "How do you improve the performance of a web application?",
        "What are CSS preprocessors, and why are they used?",
        "What is the Document Object Model (DOM), and how do you manipulate it?",
        "How do you ensure accessibility in web applications?"
    ],

    "back end developer": [
        "What are the main differences between SQL and NoSQL databases?",
        "How do you handle API authentication and authorization?",
        "What is the difference between synchronous and asynchronous programming?",
        "How do you ensure the scalability of back-end systems?",
        "What is the purpose of caching, and how do you implement it in back-end applications?"
    ],

    "computer vision": [
        "What is the difference between image classification and object detection?",
        "What are convolutional neural networks (CNNs), and how are they used in computer vision?",
        "Explain the process of image segmentation and its use cases.",
        "How do you handle image preprocessing for deep learning models?",
        "What is the role of transfer learning in computer vision tasks?"
    ],

    "artificial intelligence": [
        "What are the key differences between narrow AI and general AI?",
        "Explain the concept of reinforcement learning and its applications.",
        "What is a Markov Decision Process (MDP) in AI?",
        "How do AI systems handle decision-making under uncertainty?",
        "What are expert systems, and how do they function in AI?"
    ],

    "machine learning": [
        "What is supervised learning, and how does it differ from unsupervised learning?",
        "What is the bias-variance trade-off in machine learning?",
        "What is feature engineering, and why is it important?",
        "Explain the difference between bagging and boosting in ensemble learning.",
        "How does cross-validation help in assessing machine learning models?"
    ],
    "cloud computing": [
        "What are the different types of cloud services (IaaS, PaaS, SaaS)?",
        "What is cloud deployment, and what are the different models (private, public, hybrid)?",
        "How do you ensure the security of cloud-based applications?",
        "What is serverless computing, and when should it be used?",
        "Explain the concept of cloud scalability and elasticity."
    ],

    "devops": [
        "What is DevOps, and how does it improve the software development lifecycle?",
        "Explain the concept of Continuous Integration and Continuous Deployment (CI/CD).",
        "What are containerization and Docker, and how are they used in DevOps?",
        "How do you monitor and manage system performance in a DevOps environment?",
        "What is the role of Infrastructure as Code (IaC) in DevOps?"
    ],

    "blockchain": [
        "What is blockchain technology, and how does it work?",
        "Explain the difference between public and private blockchains.",
        "What is a consensus algorithm, and why is it important in blockchain networks?",
        "How does a smart contract work in a blockchain ecosystem?",
        "What are the key security considerations in blockchain development?"
    ],

    "big data": [
        "What are the 3 Vs of big data?",
        "Explain the differences between structured and unstructured data.",
        "How do you process large-scale data using Hadoop or Spark?",
        "What are the advantages of using NoSQL databases for big data?",
        "How do you ensure data quality in big data environments?"
    ],

    "cybersecurity": [
        "What is the difference between encryption and hashing?",
        "What are the different types of security attacks (e.g., DDoS, SQL injection)?",
        "What is multi-factor authentication (MFA), and how does it improve security?",
        "Explain the role of firewalls in network security.",
        "What is the concept of Zero Trust security?"
    ],

    "iot": [
        "What is the Internet of Things (IoT), and what are its key components?",
        "What are the different types of IoT communication protocols?",
        "How do you ensure the security of IoT devices?",
        "What is edge computing, and how is it used in IoT?",
        "What are some real-world applications of IoT in industries like healthcare or manufacturing?"
    ],

    "nlp": [
        "What is NLP, and what are some common NLP tasks?",
        "Explain the concept of word embeddings, such as Word2Vec or GloVe.",
        "What is the difference between stemming and lemmatization?",
        "How does Named Entity Recognition (NER) work in NLP?",
        "What is sentiment analysis, and how is it used in NLP applications?"
    ]

}

# --- Generate Technical Questions ---
if st.session_state["info_collected"]:
    # Normalize and process tech stack (case-insensitive matching)
    tech_stack = [
        tech.strip().lower() for tech in st.session_state["candidate_info"]["Tech Stack"] if tech.strip()
    ]

    if "questions" not in st.session_state:
        st.session_state["questions"] = {}

    # Fetch questions dynamically for each technology from the question bank
    for tech in tech_stack:
        if tech not in st.session_state["questions"]:
            # Check if the technology is in the predefined question bank
            if tech in QUESTION_BANK:
                # Format questions as a numbered list
                formatted_questions = "\n".join(
                    [f"{i + 1}. {q}" for i, q in enumerate(QUESTION_BANK[tech])]
                )
                st.session_state["questions"][tech] = formatted_questions
            else:
                # Handle cases where technology is not in the question bank
                st.session_state["questions"][tech] = (
                    f"No predefined questions are available for {tech.capitalize()}."
                )

    # Display generated questions
    st.write("### Technical Questions:")
    for tech, q in st.session_state["questions"].items():
        st.markdown(f"#### {tech.capitalize()}")
        st.markdown(q)

# --- Chatbot Interaction ---
st.write("### Chat with TalentScout Assistant")

# --- Predefined Questions with Interactive Logic ---
predefined_questions = {
    "What is the hiring process?": lambda: "Our hiring process consists of the following steps: 1. Resume Screening, 2. Initial Interview, 3. Technical Round, 4. HR Round, and 5. Final Offer.",
    "What roles are currently open?": lambda: "You can find the list of open roles on our careers page or let me know your area of interest, and I’ll guide you.",
    "What are the eligibility criteria?": lambda: "Eligibility criteria depend on the role. Generally, they include relevant educational qualifications, skills, and work experience.",
    "How can I apply for a job?": lambda: "You can apply through our careers portal by submitting your resume and filling out the application form.",
    "What is the status of my application?": lambda: ask_for_application_status(),
    "What are the interview stages?": lambda: "Our interview stages include an initial screening, technical interviews, and an HR discussion."
}

def ask_for_application_status():
    """Interactive response for application status query."""
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown("Could you please provide your application ID or registered email so I can check the status?")
    application_id = st.text_input("Enter Application ID or Email:")

    if application_id:
        # Mock response for application status
        # In a real-world scenario, fetch details from a database or API
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(f"The application with ID/email '{application_id}' is currently under review. Please check back in 2-3 days.")
        return f"The application with ID/email '{application_id}' is currently under review."
    else:
        return "Please provide a valid application ID or email to check the status."

# --- Display Predefined Questions ---
st.write("#### Frequently Asked Questions:")
selected_question = st.selectbox(
    "Choose a question:",
    ["Select a question"] + list(predefined_questions.keys()) + ["Other Query"]
)

# --- Handle User Selection as Chat ---
if selected_question != "Select a question":
    if selected_question in predefined_questions:
        # Simulate user selecting a question
        user_prompt = selected_question
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_prompt)

        # Bot responds dynamically based on the predefined question logic
        bot_response = predefined_questions[selected_question]()
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(bot_response)

    elif selected_question == "Other Query":
        # Handle custom input from the user
        if prompt := st.chat_input(f"Hi {st.session_state['candidate_info'].get('Name', 'there')}, how can I assist you?"):
            # Add Exit Functionality
            exit_keywords = ["exit", "goodbye", "quit", "bye", "okay", "thankyou"]
            if any(word in prompt.lower() for word in exit_keywords):
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    st.markdown("Thank you for interacting with TalentScout.")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)

            # --- Fallback Mechanism: Local Check for Purpose ---
            recruitment_keywords = [
                "hiring", "recruitment", "job", "candidate", "position",
                "interview", "technical", "tech stack", "skills", "experience"
            ]
            tech_keywords = [
                "Python", "Django", "Java", "Machine Learning", "AI", "Artificial Intelligence", "NLP", "Programming Language",
                "development", "coding", "framework", "software", "questions", "SQL", "Database"
            ]

            # Check if user input is relevant
            if any(word.lower() in prompt.lower() for word in recruitment_keywords + tech_keywords):
                # Chatbot response
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        # Process the prompt with OpenAI only if it passes the local check
                        for response in openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=st.session_state.messages,
                                stream=True,
                        ):
                            full_response += response.choices[0].delta.get("content", "")
                            message_placeholder.markdown(full_response + "|")
                    except Exception:
                        full_response = (
                            "I’m sorry, I couldn’t process that. Could you please clarify? Let’s focus on recruitment-related queries."
                        )
                    message_placeholder.markdown(full_response)
            else:
                # Fallback response for irrelevant inputs
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    fallback_response = (
                        "I’m here to assist with recruitment-related topics like technical questions, "
                        "candidate information, or hiring assistance. Could you clarify your query?"
                    )
                    st.markdown(fallback_response)
                    full_response = fallback_response

                # Append assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_chat_history(st.session_state.messages)

# --- End the Conversation ---
st.write("If you are done, simply type 'exit' or 'goodbye' to end the conversation.")