import streamlit as st
from litellm import completion
import os, json, difflib
from datetime import datetime
from dotenv import load_dotenv, set_key, get_key
from stlite_sandbox import stlite_sandbox

def load_env_variables():
    load_dotenv()
    if api_key := os.getenv("GROQ_API_KEY"):
        st.session_state["GROQ_API_KEY"] = api_key
    if 'stlite_refresh' not in st.session_state:
        st.session_state.stlite_refresh = False
    
    # Load API keys from environment variables
    for provider in ["GROQ", "OPENAI", "ANTHROPIC", "GOOGLE", "COHERE"]:
        env_key = f"{provider}_API_KEY"
        if api_key := os.getenv(env_key):
            st.session_state[env_key] = api_key

def load_css():
    st.markdown("""
    <style>

    </style>
    """, unsafe_allow_html=True)

def get_llm_response(prompt, history, selected_project):
    system_prompt = """You are a Streamlit app development AI assistant. Follow these rules:
1. Enclose entire app.py content within +++ delimiters.
2. Provide one code block per response.
3. Focus on efficiency without unnecessary conversation.
4. Consider the entire file when making changes.
5. Explain changes briefly after the code block.
6. List all required external packages at the end, prefixed with 'REQUIREMENTS:'.

Example response:
+++
import streamlit as st
import pandas as pd

def main():
    st.title("Data Viewer")
    data = pd.read_csv("data.csv")
    st.dataframe(data)

if __name__ == "__main__":
    main()
+++
Added pandas for data handling and created a simple data viewer.

REQUIREMENTS:
- streamlit
- pandas
"""
    project_path = os.path.join("projects", selected_project, "app.py")
    current_file_content = open(project_path).read() if os.path.exists(project_path) else ""
    file_context = f"Current app.py content:\n\n+++\n{current_file_content}\n+++\n\nMake changes based on the user's request. Or respond with typical response ignoring the code content."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": file_context},
        *[{"role": m["role"], "content": m["content"]} for m in history],
        {"role": "user", "content": prompt}
    ]

    # Get the current provider from the session state
    provider = st.session_state.get('LLM_PROVIDER', 'GROQ').upper()
    env_path = ".env"
    env_key = f"{provider}_API_KEY"
    
    # Get the API key from the environment variables
    env_api_key = get_key(env_path, env_key)
    os.environ[env_key] = env_api_key

    if not env_api_key:
        raise ValueError(f"No API key found for provider {provider}. Please check your .env file.")

    model = f"{provider.lower()}/{st.session_state.get('LLM_MODEL', 'llama-3.1-70b-versatile')}".lower()
    
    return completion(model=model, messages=messages, stream=True, api_key=env_api_key)

def save_settings(provider, api_key, model):
    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, "w").close()
    load_dotenv(env_path)

    # Update or add the API key for the given provider
    # set_key updates the in-memory env vars and writes to the .env file
    env_key = f"{provider.upper()}_API_KEY"
    st.write(env_key)
    set_key(env_path, env_key, api_key)
    st.write(api_key)
    # Update settings.json with non-sensitive information
    settings = {
        "LLM_PROVIDER": provider,
        "LLM_MODEL": model
    }
    with open("settings.json", "w") as f:
        json.dump(settings, f)

    # Update session state
    st.session_state.update(settings)
    st.session_state[env_key] = api_key

    # Reload environment variables
    load_dotenv(env_path)

def load_settings():
    load_dotenv()
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f:
            settings = json.load(f)
        st.session_state.update(settings)
    else:
        st.session_state.update({
            "LLM_PROVIDER": "groq",
            "LLM_MODEL": "llama-3.1-70b-versatile"
        })
    
    # Load API keys from environment variables
    provider = st.session_state.get('LLM_PROVIDER', 'GROQ').upper()
    env_path = ".env"
    env_key = f"{provider}_API_KEY"
    
    # Get the API key from the environment variables
    env_api_key = get_key(env_path, env_key)
    st.session_state[env_key] = env_api_key

    st.session_state.setdefault("messages", [])

def create_project(name):
    project_path = os.path.join("projects", name)
    os.makedirs(project_path, exist_ok=True)
    main_template = f"""
import streamlit as st

st.set_page_config(page_title="{name}", layout="wide", initial_sidebar_state="expanded")

def main():
    st.title("Welcome to {name}")
    st.write("This is a basic Streamlit app template.")
    if user_input := st.text_input("Enter your name"):
        st.write(f"Hello, {{user_input}}!")

if __name__ == "__main__":
    main()
"""
    with open(os.path.join(project_path, "app.py"), "w") as f:
        f.write(main_template)
    json.dump({"name": name, "created_date": datetime.now().isoformat()}, open(os.path.join(project_path, "project_notes.json"), "w"))
    return project_path

def get_projects():
    return [d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))]

def read_requirements(project_path):
    req_path = os.path.join(project_path, "requirements.txt")
    return list(set(open(req_path).read().splitlines()) if os.path.exists(req_path) else [])

def update_requirements(project_path, new_imports):
    existing_reqs = set(read_requirements(project_path))
    new_reqs = set(new_imports) - existing_reqs - {"streamlit"}
    if new_reqs:
        with open(os.path.join(project_path, "requirements.txt"), "a") as f:
            for req in new_reqs:
                f.write(f"{req}\n")
        st.toast(f"Added new requirements: {', '.join(new_reqs)}")

def extract_imports(code):
    import ast
    tree = ast.parse(code)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split('.')[0])
    return imports

def parse_ai_response(response):
    code_start = response.index("+++")
    code_end = response.index("+++", code_start + 3)
    code = response[code_start + 3:code_end].strip()
    imports = list(extract_imports(code))
    requirements_start = response.find("REQUIREMENTS:")
    if requirements_start != -1:
        imports.extend([req.strip('- ').split('==')[0] for req in response[requirements_start:].split('\n')[1:] if req.strip()])
    return code, imports

@st.fragment
def update_code_confirmation(full_response):
    new_code, new_imports = parse_ai_response(full_response)

    if st.button("Confirm"):
        save_code(st.session_state.selected_project, new_code)
        # update_requirements(os.path.join("projects", st.session_state.selected_project), new_imports)
        st.rerun()
        
    if st.button("Revert"):
        st.rerun()
        
def save_code(project, code):
    try:
        with open(os.path.join("projects", project, "app.py"), "w") as f:
            f.write(code)
        return True
    except Exception as e:
        st.error(f"Error saving code: {str(e)}")
        return False

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

def format_response(response):
    # This function ensures that the +++ delimiters are properly placed
    if "+++" not in response:
        return response
    
    parts = response.split("+++")
    formatted_parts = []
    for i, part in enumerate(parts):
        if i == 0:
            formatted_parts.append(part.strip())
        elif i % 2 == 1:  # Code part
            formatted_parts.append(f"+++\n{part.strip()}\n+++")
        else:  # Text part
            formatted_parts.append(part.strip())
    
    return "\n\n".join(formatted_parts)

def chat_page():
    st.title("💬 Chat")
    col1, col2 = st.columns([2, 3])
    with col1:
        with st.container(border=False, height=750):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    content = message["content"]
                    if "+++" in content:
                        parts = content.split("+++")
                        for i, part in enumerate(parts):
                            if i % 2 == 0:
                                st.markdown(part.strip())
                            else:
                                st.code(part.strip(), language="python")
                    else:
                        st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
            code_placeholder = st.empty()
    with col2:
        preview_tab, code_tab = st.tabs(["Preview", "Code"])
        with preview_tab:
            st.subheader("Preview")
            if st.session_state.selected_project:
                project_path = os.path.join("projects", st.session_state.selected_project)
                app_path = os.path.join(project_path, "app.py")
                if os.path.exists(app_path):
                    code_content = open(app_path).read()
                    requirements = read_requirements(project_path)  # This now returns a list
                    stlite_sandbox(code_content, height=500, editor=False, border=False, requirements=requirements, scrollable=True)
                else:
                    st.warning(f"No app.py found in {st.session_state.selected_project} project.")
            else:
                st.info("Please select a project to preview.")
        with code_tab:
            code_editor(st.session_state.selected_project)
                
    if prompt := st.chat_input("What's your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        full_response = ""
        
        for response in get_llm_response(prompt, st.session_state.messages[:-1], st.session_state.selected_project):
            if isinstance(response, str):
                full_response += response
            elif hasattr(response, 'choices') and response.choices:
                delta = response.choices[0].delta
                if delta and delta.content:
                    full_response += delta.content
            
            # Format the response as it's being generated
            formatted_response = format_response(full_response)
            with code_placeholder.container(height=750, border=False):
                with st.chat_message("assistant"):
                    if "+++" in formatted_response:
                        parts = formatted_response.split("+++")
                        for i, part in enumerate(parts):
                            if i % 2 == 0:
                                st.markdown(part.strip())
                            else:
                                st.code(part.strip(), language="python")
                    else:
                        st.markdown(formatted_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        if "+++" in full_response:
            update_code_confirmation(full_response)

def projects_page():
    st.title("📁 Projects")
    os.makedirs("projects", exist_ok=True)
    projects = get_projects()
    if not projects:
        st.warning("No projects found.")
    with st.expander("Create New Project"):
        with st.form("create_project", border=False):
            new_project_name = st.text_input("Project Name")
            if st.form_submit_button("Create Project") and new_project_name:
                create_project(new_project_name)
                st.toast(f"Project '{new_project_name}' created successfully!")
    for project in projects:
        with st.expander(project):
            project_path = os.path.join("projects", project)
            notes_path = os.path.join(project_path, "project_notes.json")
            if os.path.exists(notes_path):
                notes = json.load(open(notes_path))
                st.write(f"Created: {notes['created_date']}")
            st.write("📄 app.py")

def settings_page():
    st.title("⚙️ Settings")
    llm_provider = st.selectbox("LLM Provider", ["Groq", "OpenAI", "Anthropic", "Google", "Cohere"], index=0)
    api_key = st.text_input(f"{llm_provider} API Key", type="password", value=st.session_state.get(f"{llm_provider.upper()}_API_KEY", ""))
    model = st.text_input("Model", value=st.session_state.get("LLM_MODEL", "llama-3.1-70b-versatile"))
    if st.button("Save Settings"):
        save_settings(llm_provider, api_key, model)
        st.toast("Settings saved successfully!")

def main():
    st.set_page_config(page_title="Streamlit LLM Chat", page_icon="💬", layout="wide", initial_sidebar_state="expanded")
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
        st.navigation([
            st.Page(chat_page, title="Chat", icon="💬"),
            st.Page(projects_page, title="Projects", icon="📁"),
            st.Page(settings_page, title="Settings", icon="⚙️")
        ]).run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()