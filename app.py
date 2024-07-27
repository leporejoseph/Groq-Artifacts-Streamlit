import streamlit as st
from litellm import completion
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from stlite_sandbox import stlite_sandbox
import subprocess
import sys

def load_env_variables():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        st.session_state["GROQ_API_KEY"] = groq_api_key

# Utility functions
def load_css():
    st.markdown("""
    <style>
    /* Existing styles ... */

    /* Chat input styling */
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #ffffff;
    }

    /* Send button styling */
    .stButton > button {
        background-color: #2d2d2d;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }

    /* Adjust column widths */
    .main .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e1e1e;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }

    .stChatMessage .content p {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

def get_llm_response(prompt, history, selected_project):
    system_prompt = """You are a helpful AI assistant specialized in Streamlit app development.

ADDITIONAL INSTRUCTIONS:

1. When providing code changes or new code for the app.py file, always enclose the entire file content within +++ delimiters. For example:

+++
import streamlit as st

def main():
    st.title("Hello, World!")

if __name__ == "__main__":
    main()
+++

2. Provide only one code block per response.
3. Focus on completing the task efficiently without unnecessary conversation. Avoid phrases like "Certainly!" or "Is there anything else I can help you with?"
4. When making changes to existing code, consider the entire file content and provide the full updated file within the +++ delimiters.
5. After providing a code block, briefly explain the changes or additions you've made.
"""

    # Read the current content of app.py
    current_file_content = ""
    if selected_project:
        project_path = os.path.join("projects", selected_project, "app.py")
        if os.path.exists(project_path):
            with open(project_path, "r") as file:
                current_file_content = file.read()

    # Add the current file content to the prompt
    file_context = f"Here is the current content of app.py:\n\n+++\n{current_file_content}\n+++\n\nPlease make changes or additions based on the user's request."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": file_context},
        *[{"role": m["role"], "content": m["content"]} for m in history],
        {"role": "user", "content": prompt}
    ]

    model_formatted = (f"{st.session_state.get('LLM_PROVIDER', 'groq')}/{st.session_state.get('LLM_MODEL', 'llama-3.1-70b-versatile')}").lower()
    response = completion(
        model=model_formatted,
        messages=messages,
        stream=True,
    )

    for chunk in response:
        yield chunk.choices[0].delta.content or ""

def save_settings(provider, api_key, model):
    settings = {
        "LLM_PROVIDER": provider,
        "LLM_MODEL": model,
        f"{provider.upper()}_API_KEY": api_key
    }
    with open("settings.json", "w") as f:
        json.dump(settings, f)
    st.session_state.update(settings)

def load_settings():
    load_env_variables()
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f:
            settings = json.load(f)
        st.session_state.update(settings)
    else:
        st.session_state.update({
            "LLM_PROVIDER": "groq",
            "LLM_MODEL": "llama-3.1-70b-versatile"
        })
    # Use the GROQ_API_KEY from .env if it's not in settings.json
    if "GROQ_API_KEY" not in st.session_state:
        st.session_state["GROQ_API_KEY"] = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "edited_code" not in st.session_state:
        st.session_state.edited_code = ""
    if "project_create" not in st.session_state:
        st.session_state.project_create = False

def test_llm_connection():
    try:
        model_formatted=(f"{st.session_state.get('LLM_PROVIDER', 'groq')}/{st.session_state.get('LLM_MODEL', 'llama-3.1-70b-versatile')}").lower()
        completion(
            model=model_formatted,
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        st.error(f"Error testing LLM connection: {e}")
        return False

def create_project(name):
    project_path = os.path.join("projects", name)
    os.makedirs(project_path, exist_ok=True)
    
    # Create main app.py
    main_template = f"""
import streamlit as st

st.set_page_config(page_title="{name}", layout="wide", initial_sidebar_state="expanded")

def main():
    st.title("Welcome to {name}")
    st.write("This is a basic Streamlit app template.")

    # Add a sample widget
    user_input = st.text_input("Enter your name")
    if user_input:
        st.write(f"Hello, {{user_input}}!")

if __name__ == "__main__":
    main()
"""
    with open(os.path.join(project_path, "app.py"), "w") as f:
        f.write(main_template)
    
    # Create project_notes.json
    project_notes = {
        "name": name,
        "created_date": datetime.now().isoformat()
    }
    with open(os.path.join(project_path, "project_notes.json"), "w") as f:
        json.dump(project_notes, f)

    return project_path

def get_projects():
    projects_dir = "projects"
    return [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]

@st.fragment
def code_editor(selected_project):
    if selected_project:
        project_path = os.path.join("projects", selected_project, "app.py")
        if os.path.exists(project_path):
            with open(project_path, "r") as file:
                original_code = file.read()
            
            edited_code = st.text_area(
                "Edit Code",
                value=original_code,
                height=800,
                key=f"edited_code_{selected_project}"
            )

            # Check if the code has been edited
            if edited_code != original_code:
                save_code(selected_project, edited_code)
                st.rerun()
        else:
            st.warning(f"No app.py file found in the {selected_project} project.")
    else:
        st.info("Please select a project to view and edit its code.")
        
# Page functions
def chat_page():
    st.title("💬 Chat")

    col1, col2 = st.columns([2, 3])

    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        code_placeholder = st.empty()

    with col2:
        preview_tab, code_tab = st.tabs(["Preview", "Code"])

        with preview_tab:
            st.subheader("Preview")
            if st.session_state.selected_project:
                project_path = os.path.join("projects", st.session_state.selected_project, "app.py")
                if os.path.exists(project_path):
                    with open(project_path, "r") as file:
                        code_content = file.read()
                    stlite_sandbox(code_content, height=500, editor=False)
                else:
                    st.warning(f"No app.py file found in the {st.session_state.selected_project} project.")
            else:
                st.info("Please select a project to preview.")

        with code_tab:
            st.subheader("Code")
            if st.session_state.selected_project:
                project_path = os.path.join("projects", st.session_state.selected_project, "app.py")
                if os.path.exists(project_path):
                    with open(project_path, "r") as file:
                        code_content = file.read()
                    st.code(code_content, language="python")
                else:
                    st.warning(f"No app.py file found in the {st.session_state.selected_project} project.")
            else:
                st.info("Please select a project to view its code.")

    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        full_response = ""
        for response in get_llm_response(prompt, st.session_state.messages[:-1], st.session_state.selected_project):
            full_response += response
        
        # Process the response to handle code blocks
        if "+++" in full_response:
            code_start = full_response.index("+++")
            code_end = full_response.index("+++", code_start + 3)
            code_block = full_response[code_start:code_end + 3]
            
            with code_placeholder.form("update_code_form", border=False):
                st.code(code_block, language="python")
                if st.form_submit_button("Confirm"):
                    save_code(st.session_state.selected_project, code_block[3:-3])
                
        st.session_state.messages.append({"role": "assistant", "content": full_response.replace("+++", "")})

def save_code(project, code):
    try:
        project_path = os.path.join("projects", project, "app.py")
        with open(project_path, "w") as file:
            file.write(code)
        return True
    except Exception as e:
        st.error(f"An error occurred while saving the code: {str(e)}")
        return False


def projects_page():
    st.title("📁 Projects")

    if not os.path.exists("projects"):
        os.makedirs("projects")

    projects = get_projects()

    if not projects:
        st.warning("No projects found.")

    with st.expander("Create New Project"):
        with st.form("create_project"):
            new_project_name = st.text_input("Project Name")
            submit_button = st.form_submit_button("Create Project")
            if submit_button and new_project_name:
                create_project(new_project_name)
                st.toast(f"Project '{new_project_name}' created successfully!")

    for project in projects:
        with st.expander(project):
            project_path = os.path.join("projects", project)
            notes_path = os.path.join(project_path, "project_notes.json")
            if os.path.exists(notes_path):
                with open(notes_path, "r") as f:
                    notes = json.load(f)
                st.write(f"Created: {notes['created_date']}")
            st.write(f"📄 app.py")

def settings_page():
    st.title("⚙️ Settings")

    llm_provider = st.selectbox("LLM Provider", ["Groq", "OpenAI", "Anthropic", "Google", "Cohere"], index=0)
    
    if llm_provider == "Groq":
        api_key = st.text_input("Groq API Key", type="password", value=st.session_state.get("GROQ_API_KEY", ""))
    else:
        api_key = st.text_input(f"{llm_provider} API Key", type="password", value=st.session_state.get(f"{llm_provider.upper()}_API_KEY", ""))
    
    model = st.text_input("Model", value=st.session_state.get("LLM_MODEL", "llama-3.1-70b-versatile"))

    if st.button("Save Settings"):
        save_settings(llm_provider, api_key, model)
        st.success("Settings saved successfully!")

        if test_llm_connection():
            st.success("LLM connection tested successfully!")
        else:
            st.error("Failed to connect to LLM. Please check your settings and API key.")

# Main app
def main():
    st.set_page_config(
        page_title="Streamlit LLM Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.streamlit.io/community',
            'Report a bug': "https://github.com/yourusername/yourrepository/issues",
            'About': "# Streamlit LLM Chat\nThis is a Streamlit app for chatting with an LLM and managing projects."
        }
    )
    
    load_css()
    load_settings()

    try:
        with st.sidebar:
            projects = get_projects()
            if projects:
                st.selectbox("Select a project", projects, key="selected_project")
            else:
                st.warning("No projects found. Create a new project in the Projects page.")
                st.session_state.selected_project = None

        pages = st.navigation([
            st.Page(chat_page, title="Chat", icon="💬"),
            st.Page(projects_page, title="Projects", icon="📁"),
            st.Page(settings_page, title="Settings", icon="⚙️")
        ])
        
        pages.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()