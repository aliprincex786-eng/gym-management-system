"""
FitTrack — Gym Management System
Streamlit UI + SQLite storage + AI Assistant powered by openai/gpt-oss-120b (via Groq's
OpenAI-compatible API). Swap GROQ_API_KEY / API_BASE_URL in secrets to use another
provider that hosts the same model (e.g. OpenRouter, Together).
"""

import os
import sqlite3
from datetime import datetime, date

import pandas as pd
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="FitTrack Gym Manager", page_icon="🏋️", layout="wide")

DB_PATH = "gym.db"
MEMBERSHIP_PLANS = {"Basic": 20.0, "Standard": 35.0, "Premium": 50.0}

# ------------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            join_date TEXT,
            membership_type TEXT,
            status TEXT DEFAULT 'Active'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            check_in TEXT,
            check_out TEXT,
            FOREIGN KEY(member_id) REFERENCES members(id) ON DELETE CASCADE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            amount REAL,
            payment_date TEXT,
            method TEXT,
            plan TEXT,
            next_due_date TEXT,
            FOREIGN KEY(member_id) REFERENCES members(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ------------------------------------------------------------------
# DATA HELPERS  (all use parameterized queries — no string-built SQL)
# ------------------------------------------------------------------
def add_member(name, phone, email, membership_type):
    conn = get_conn()
    conn.execute(
        "INSERT INTO members (name, phone, email, join_date, membership_type, status) "
        "VALUES (?, ?, ?, ?, ?, 'Active')",
        (name, phone, email, date.today().isoformat(), membership_type),
    )
    conn.commit()
    conn.close()


def get_members(active_only=False):
    conn = get_conn()
    q = "SELECT * FROM members"
    if active_only:
        q += " WHERE status = 'Active'"
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df


def update_member_status(member_id, status):
    conn = get_conn()
    conn.execute("UPDATE members SET status = ? WHERE id = ?", (status, member_id))
    conn.commit()
    conn.close()


def delete_member(member_id):
    conn = get_conn()
    conn.execute("DELETE FROM members WHERE id = ?", (member_id,))
    conn.commit()
    conn.close()


def check_in(member_id):
    conn = get_conn()
    conn.execute(
        "INSERT INTO attendance (member_id, check_in) VALUES (?, ?)",
        (member_id, datetime.now().isoformat(timespec="minutes")),
    )
    conn.commit()
    conn.close()


def check_out(attendance_id):
    conn = get_conn()
    conn.execute(
        "UPDATE attendance SET check_out = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="minutes"), attendance_id),
    )
    conn.commit()
    conn.close()


def get_attendance():
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT a.id, m.name, a.check_in, a.check_out
           FROM attendance a JOIN members m ON a.member_id = m.id
           ORDER BY a.id DESC""",
        conn,
    )
    conn.close()
    return df


def add_payment(member_id, amount, method, plan, next_due_date):
    conn = get_conn()
    conn.execute(
        "INSERT INTO payments (member_id, amount, payment_date, method, plan, next_due_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (member_id, amount, date.today().isoformat(), method, plan, next_due_date),
    )
    conn.commit()
    conn.close()


def get_payments():
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT p.id, m.name, p.amount, p.payment_date, p.method, p.plan, p.next_due_date
           FROM payments p JOIN members m ON p.member_id = m.id
           ORDER BY p.payment_date DESC""",
        conn,
    )
    conn.close()
    return df


# ------------------------------------------------------------------
# AI ASSISTANT  (openai/gpt-oss-120b via Groq's OpenAI-compatible endpoint)
# ------------------------------------------------------------------
@st.cache_resource
def get_ai_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    base_url = st.secrets.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def ask_ai(question, context_summary):
    client = get_ai_client()
    if client is None:
        return ("⚠️ No API key configured. Add GROQ_API_KEY to your Streamlit secrets "
                "to enable the AI assistant.")
    system_prompt = (
        "You are a helpful gym-operations assistant for a fitness studio. "
        "Use the gym summary data below when relevant. Keep answers concise and practical.\n\n"
        f"GYM SUMMARY:\n{context_summary}"
    )
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI request failed: {e}"


def build_context_summary():
    members = get_members()
    payments = get_payments()
    active = (members["status"] == "Active").sum() if not members.empty else 0
    total_members = len(members)
    revenue = payments["amount"].sum() if not payments.empty else 0
    return (
        f"Total members: {total_members}\n"
        f"Active members: {active}\n"
        f"Total revenue recorded: ${revenue:,.2f}\n"
        f"Membership plans: {', '.join(MEMBERSHIP_PLANS.keys())}"
    )


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.sidebar.title("🏋️ FitTrack")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Members", "Attendance", "Payments", "AI Assistant"],
)

if page == "Dashboard":
    st.title("Dashboard")
    members = get_members()
    payments = get_payments()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total members", len(members))
    col2.metric("Active members", int((members["status"] == "Active").sum()) if not members.empty else 0)
    col3.metric("Total revenue", f"${payments['amount'].sum():,.2f}" if not payments.empty else "$0.00")

    st.subheader("Recent attendance")
    st.dataframe(get_attendance().head(10), use_container_width=True)

elif page == "Members":
    st.title("Members")

    with st.expander("➕ Add new member"):
        with st.form("add_member_form", clear_on_submit=True):
            name = st.text_input("Full name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            plan = st.selectbox("Membership type", list(MEMBERSHIP_PLANS.keys()))
            submitted = st.form_submit_button("Add member")
            if submitted:
                if name.strip():
                    add_member(name.strip(), phone.strip(), email.strip(), plan)
                    st.success(f"Added {name}.")
                    st.rerun()
                else:
                    st.error("Name is required.")

    members = get_members()
    st.dataframe(members, use_container_width=True)

    if not members.empty:
        st.subheader("Update a member")
        mid = st.selectbox(
            "Select member",
            members["id"],
            format_func=lambda x: members.loc[members["id"] == x, "name"].values[0],
        )
        c1, c2 = st.columns(2)
        with c1:
            new_status = st.selectbox("Status", ["Active", "Inactive", "Frozen"])
            if st.button("Update status"):
                update_member_status(mid, new_status)
                st.success("Status updated.")
                st.rerun()
        with c2:
            if st.button("🗑️ Delete member", type="secondary"):
                delete_member(mid)
                st.success("Member deleted.")
                st.rerun()

elif page == "Attendance":
    st.title("Attendance")
    members = get_members(active_only=True)

    if members.empty:
        st.info("Add an active member first.")
    else:
        mid = st.selectbox(
            "Member",
            members["id"],
            format_func=lambda x: members.loc[members["id"] == x, "name"].values[0],
        )
        if st.button("✅ Check in"):
            check_in(mid)
            st.success("Checked in.")
            st.rerun()

    st.subheader("Log")
    log = get_attendance()
    st.dataframe(log, use_container_width=True)

    open_sessions = log[log["check_out"].isna()] if not log.empty else log
    if not open_sessions.empty:
        st.subheader("Check out")
        aid = st.selectbox(
            "Open session",
            open_sessions["id"],
            format_func=lambda x: f"{open_sessions.loc[open_sessions['id']==x,'name'].values[0]} "
                                   f"({open_sessions.loc[open_sessions['id']==x,'check_in'].values[0]})",
        )
        if st.button("🚪 Check out"):
            check_out(aid)
            st.success("Checked out.")
            st.rerun()

elif page == "Payments":
    st.title("Payments")
    members = get_members()

    if members.empty:
        st.info("Add a member first.")
    else:
        with st.form("payment_form", clear_on_submit=True):
            mid = st.selectbox(
                "Member",
                members["id"],
                format_func=lambda x: members.loc[members["id"] == x, "name"].values[0],
            )
            plan = st.selectbox("Plan", list(MEMBERSHIP_PLANS.keys()))
            amount = st.number_input("Amount ($)", min_value=0.0, value=MEMBERSHIP_PLANS[plan], step=1.0)
            method = st.selectbox("Method", ["Cash", "Card", "Bank transfer", "Other"])
            next_due = st.date_input("Next due date")
            submitted = st.form_submit_button("Record payment")
            if submitted:
                add_payment(mid, amount, method, plan, next_due.isoformat())
                st.success("Payment recorded.")
                st.rerun()

    st.subheader("Payment history")
    st.dataframe(get_payments(), use_container_width=True)

elif page == "AI Assistant":
    st.title("AI Assistant")
    st.caption("Powered by openai/gpt-oss-120b")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask about your gym (revenue, retention tips, schedules, etc.)")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_ai(question, build_context_summary())
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))
