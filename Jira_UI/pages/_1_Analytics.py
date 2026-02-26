import streamlit as st
from Jira_UI import run_agent
import re
from datetime import datetime

st.set_page_config(
    page_title="SMART-QA Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown('<h1 style="color:#4A90E2;">📊 SMART-QA Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:gray;">View Jira & Robot Framework metrics in real-time</p>', unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {
        "total_passed": 0,
        "total_failed": 0,
        "bugs_created": 0
    }

# -----------------------------
# Run automation button
# -----------------------------
if st.button("▶ Run Automation & Fetch Metrics"):
    with st.spinner("Running Robot Framework tests and fetching Jira data..."):
        response_list = run_agent("run tests")  # returns list of tuples

    # Extract tool output (Robot Framework summary)
    robot_reports = [msg for role, msg in response_list if role == "tool"]

    total_passed = 0
    total_failed = 0
    bugs_created = 0

    for report in robot_reports:
        # Example report: "Robot Results - Passed: 30, Failed: 9. Created Bug: SCRUM-101."
        # Safe regex parsing
        match_passed = re.search(r"Passed:\s*(\d+)", report)
        match_failed = re.search(r"Failed:\s*(\d+)", report)
        match_bug = re.findall(r"Created Bug:\s*([A-Z0-9-]+)", report)

        passed = int(match_passed.group(1)) if match_passed else 0
        failed = int(match_failed.group(1)) if match_failed else 0
        bug_count = len(match_bug)

        total_passed += passed
        total_failed += failed
        bugs_created += bug_count

    # Update session state
    st.session_state.analytics_data["total_passed"] += total_passed
    st.session_state.analytics_data["total_failed"] += total_failed
    st.session_state.analytics_data["bugs_created"] += bugs_created

    st.success("✅ Automation executed successfully!")

# -----------------------------
# Display metrics
# -----------------------------
st.markdown("## 📈 Test Case Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("✅ Total Passed", st.session_state.analytics_data["total_passed"])
col2.metric("❌ Total Failed", st.session_state.analytics_data["total_failed"])
col3.metric("🐞 Bugs Created", st.session_state.analytics_data["bugs_created"])

# -----------------------------
# Latest reports
# -----------------------------
st.markdown("## 📝 Latest Robot Framework Reports")
if "robot_reports" in locals() and robot_reports:
    for idx, report in enumerate(robot_reports, start=1):
        st.markdown(f"**Report {idx}:**")
        st.code(report)
else:
    st.info("No automation has been run yet.")