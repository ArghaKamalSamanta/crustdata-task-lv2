import re
import os
import shutil
import time
import streamlit as st
from llm_response import generate_response
from api_validator import validate_api_call, attempt_to_fix_api_call
from additional_knowledge import create_vectorDB

# Streamlit UI
st.set_page_config(page_title="API Chatbot", layout="wide")
st.title("Crustdata API Chatbot")

# Clean up any leftovers from the previous session
leftovers = []

for item in os.listdir(os.getcwd()):
    item_path = os.path.join(os.getcwd(), item)
    if os.path.isdir(item_path) and item.startswith("temp_"):
        leftovers.append(item_path)

if not st.session_state.get("temp_dir_checked"):
    for path in leftovers:
        shutil.rmtree(path)
    st.session_state["temp_dir_checked"] = True

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

def process_api_request(response):
    if extract_api_example(response) == None:
        return response
    
    results = extract_api_example(response)
    for result in results:
        lang = result['language']
        api_example = result['code']
        start_idx = result['start_idx']
        end_idx = result['end_idx']  

        # Show a pop-up-like message while validating the code block
        with st.spinner("Bot response contains some executables. Verifying them for you..."):
            validation_result, error_log = validate_api_call(api_example)

            if validation_result:
                continue  # API call is valid, return original response
            else:
                # Attempt to fix the API call using error logs
                complete_api_example = f"```{lang}\n{api_example}```"
                fixed_api_example = attempt_to_fix_api_call(complete_api_example, error_log)
                if "```" in fixed_api_example:
                    fixed_api_example = "\n```" + re.search(r"```(.*?)```", fixed_api_example, re.DOTALL).group(1) + "```\n"
                else:
                    fixed_api_example = complete_api_example
                
                pattern = re.escape(complete_api_example)
                match = re.search(pattern, response)
                start_idx = match.start()
                end_idx = match.end()
                response = response[:start_idx] + fixed_api_example + response[end_idx + 1:]
    return response

def extract_api_example(response):
    if "```" in response: 
        # Match code blocks enclosed in ``` followed by a language identifier
        pattern = r"```(.*?)\n(.*?)```"
        matches = list(re.finditer(pattern, response, re.DOTALL))

        results = []
        for match in matches:
            language = match.group(1).strip()  # Capture the language (e.g., python)
            code = match.group(2)             # Capture the code inside
            start_idx, end_idx = match.span() # Get the start and end indices
            results.append({
                "language": language,
                "code": code,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
        return results
    return None

# Display chat history (if any)
if st.session_state.history:
    for chat in st.session_state.history:
        with st.container():
            # User's query (right-aligned)
            st.markdown(
                f"""
                <div style="
                    display: flex; 
                    justify-content: flex-end; 
                    margin-bottom: 10px;">
                    <div style="
                        background-color: #DCF8C6; 
                        color: black; 
                        padding: 10px 15px; 
                        border-radius: 15px; 
                        max-width: 60%; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        {chat['user']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Bot's response (left-aligned)
            st.markdown(
                f"""
                <div style="
                    display: flex; 
                    justify-content: flex-start; 
                    margin-bottom: 10px;">
                    <div style="
                        background-color: #FFF9E0; 
                        color: black; 
                        padding: 10px 15px; 
                        border-radius: 15px; 
                        max-width: 60%; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <p>{chat['bot']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.container():
    if not st.session_state.history:
        st.markdown("---")

    user_input = st.text_input(
        "Ask a question about Crustdata's APIs:", 
        key="user_input", 
        value=""
    )
    
    if st.button("Submit"):
        if user_input.strip():
            response = generate_response(user_input)
            response = process_api_request(response)
            st.session_state.history.append({"user": user_input, "bot": response})
            st.rerun()

    @st.dialog("Add More Knowledge")
    def add_more_knowledge():
        uploaded_files = st.file_uploader(
            "Upload .txt files to update the knowledge base", 
            type=["txt"], 
            accept_multiple_files=True
        )
        if st.button("Add"):
            if uploaded_files:
                print("Files uploaded successfully!")  # Debugging print statement
                with st.spinner("Processing uploaded files and updating the knowledge base..."):
                    create_vectorDB(uploaded_files)
                st.success("Knowledge base updated successfully!")
                time.sleep(0.07)
                st.stop()
            else:
                st.warning("Please upload files before clicking the button.")

    if st.button("Upload More Files"):
        add_more_knowledge()