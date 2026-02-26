import streamlit as st
from Jira_UI import run_agent
from datetime import datetime
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="SMART-QA Agent",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.tool-box {
    background-color: #f4f6f9;
    padding: 12px;
    border-radius: 10px;
    border-left: 5px solid #4A90E2;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("## 🤖 SMART-QA Agent")
st.markdown("Interact with Jira & Robot Framework using AI")

# -------------------------------
# Session State Initialization
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_requests" not in st.session_state:
    st.session_state.total_requests = 0

# -------------------------------
# Sidebar Actions
# -------------------------------
st.sidebar.header("⚙️ Quick Actions")

if st.sidebar.button("🔍 List Bugs (SCRUM)"):
    user_input = "list bugs in SCRUM"
elif st.sidebar.button("▶ Run Automation"):
    user_input = "run tests"
elif st.sidebar.button("✅ Close Ticket Example"):
    user_input = "close SCRUM-1"
else:
    user_input = None

# Chat input box
chat_input = st.chat_input("Type your command here...")
if chat_input:
    user_input = chat_input

# -------------------------------
# Run Agent
# -------------------------------
if user_input:
    st.session_state.total_requests += 1
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))

    # Show user message immediately
    with st.chat_message("user"):
        st.write(f"🕒 {timestamp}")
        st.write(user_input)

    # Placeholder for AI typing
    ai_msg_container = st.chat_message("assistant")
    with ai_msg_container:
        placeholder = st.empty()
        placeholder.write("🤖 AI is typing...")

    # Call SMART-QA agent
    responses = run_agent(user_input)

    # Stream AI and tool responses
    for role, message in responses:
        if role == "ai":
            with ai_msg_container:
                placeholder.empty()
                displayed_text = ""
                for word in message.split():
                    displayed_text += word + " "
                    placeholder.markdown(displayed_text)
                    time.sleep(0.02)
                placeholder.markdown(message)

        elif role == "tool":
            st.chat_message("assistant").markdown(f"""
            <div class="tool-box">
            🔧 <b>Tool Execution Result</b><br><br>
            {message}
            </div>
            """, unsafe_allow_html=True)

        # Save AI/tool output to history
        st.session_state.chat_history.append((role, message))

# -------------------------------
# Sidebar Metrics
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Stats")
st.sidebar.metric("Total Requests", st.session_state.total_requests)
st.sidebar.markdown("Status: 🟢 Online")