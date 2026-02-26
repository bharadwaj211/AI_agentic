import streamlit as st
import os

st.title("⚙️ SMART-QA Settings")

st.markdown("Configure Jira project and Robot Framework paths.")

jira_project = st.text_input("Jira Project Key", value="SCRUM")
robot_path = st.text_input("Robot Framework Tests Path", value="tests/")
robot_output = st.text_input("Robot Output Directory", value="results")

st.markdown("---")
st.info("Changes are for session only and do not persist to .env yet.")