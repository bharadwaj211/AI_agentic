import streamlit as st
from datetime import datetime

st.title("📜 SMART-QA Activity History")

if "chat_history" in st.session_state and st.session_state.chat_history:
    for role, msg in st.session_state.chat_history:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.write(f"[{timestamp}] [{role.upper()}] {msg}")
else:
    st.info("No activity yet.")