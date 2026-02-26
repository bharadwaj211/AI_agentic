import streamlit as st

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SMART-QA Control Center",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# Welcome / Landing Page
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding:50px">
        <h1>🤖 SMART-QA Automation Control Center</h1>
        <p style="font-size:18px; color:gray;">
        AI-powered Jira & Robot Framework Automation Platform
        </p>
        <img src="https://img.icons8.com/ios-filled/100/000000/robot-2.png"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Use the sidebar to navigate between pages:\n\n"
    "- 📊 Analytics: View Jira & test metrics\n"
    "- 🤖 Agent: Chat with SMART-QA agent\n"
    "- 📜 History: Review past commands and outputs\n"
    "- ⚙️ Settings: Configure project and test paths"
)