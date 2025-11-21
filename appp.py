# app.py
import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import os
import random
from pathlib import Path

# === DOWNLOAD NLTK DATA ===
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')

# === PAGE CONFIG & THEME-LIKE CSS ===
st.set_page_config(page_title="CS Genie Pro", page_icon="🤖", layout="wide")

# --- Custom CSS for look & feel ---
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #071133 40%, #07112a 100%);
        color: #e6eef8;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Header card */
    .header {
        background: linear-gradient(90deg, rgba(99,102,241,0.12), rgba(56,189,248,0.08));
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        color: light blue;
    }
    .title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .subtitle {
        color:grey;
        margin-top: 0;
        margin-bottom: 0;
    }

    /* Sidebar */
    .css-1d391kg { /* streamlit sidebar padding class may vary; harmless fallback */
        padding-top: 10px;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(90deg, #1e293b, #0b1220);
        padding: 12px 14px;
        border-radius: 12px;
        color: #e6eef8;
        margin: 6px 0;
        border-left: 4px solid #60a5fa;
    }
    .bot-bubble {
        background: linear-gradient(90deg, #062e46, #042a35);
        padding: 12px 14px;
        border-radius: 12px;
        color: #e6eef8;
        margin: 6px 0;
        border-left: 4px solid #34d399;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#6366f1,#06b6d4);
        color: white;
        border-radius: 10px;
        padding: 6px 12px;
        box-shadow: 0 6px 14px rgba(2,6,23,0.5);
    }

    /* Small muted text */
    .muted {
        color: #9fb3d9;
        font-size: 13px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.03);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================================
# HARD-CODED MESSAGES (Option B)
# ========================================
greeting_response = "Hello! 👋 I'm **CS Genie Pro** — your friendly coding assistant. How can I help you today?"
goodbye_response = "Goodbye! 👋 Keep learning — you did great today. If you need me again, I'm right here!"
motivation_response_list = [
    "Keep going — every expert was once a beginner! 🚀",
    "Small steps every day lead to big results. 💪",
    "Code. Debug. Learn. Repeat. You got this! 🔁",
    "Mistakes are proof you're trying. Keep building! 🔧",
    "Focus on progress, not perfection. ✨"
]
# choose one quote to show on the UI
motivation_quote = random.choice(motivation_response_list)

# ========================================
# LOAD CS TERMS DATA
# ========================================
@st.cache_data
def load_data():
    # Resolve cs_terms.json relative to this script first, then fall back to cwd
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "cs_terms.json",
        Path.cwd() / "cs_terms.json",
        script_dir.parent / "cs_terms.json",
    ]

    raw_data = None
    tried = []
    for p in candidates:
        tried.append(str(p))
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            break

    if raw_data is None:
        raise FileNotFoundError(
            "Could not find 'cs_terms.json'. Paths tried: " + ", ".join(tried)
        )

    # Convert to DataFrame
    df = pd.DataFrame(raw_data)

    # Explode basic_questions and basic_answers into rows for better matching
    qa_rows = []
    for _, row in df.iterrows():
        term = row.get('term', '')
        definition = row.get('definition', '')
        if 'basic_questions' in row and 'basic_answers' in row:
            for q, a in zip(row['basic_questions'], row['basic_answers']):
                qa_rows.append({
                    'term': term,
                    'definition': definition,
                    'question': q,
                    'answer': a,
                    'type': 'qa'
                })
        # Also keep original term + definition
        qa_rows.append({
            'term': term,
            'definition': definition,
            'question': term,
            'answer': definition,
            'type': 'term'
        })

    return pd.DataFrame(qa_rows)


data = load_data()

# === TEXT CLEANING FUNCTION ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text.lower().strip()

# === PREPARE SEARCHABLE TEXT ===
data['cleaned'] = data['question'].apply(clean_text)

# Combine all searchable text
search_texts = data['cleaned'].tolist()

# Fit TF-IDF (done once at start)
vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
X = vectorizer.fit_transform(search_texts)

# ========================================
# LEFT SIDEBAR MENU (no Upload)
# ========================================
st.sidebar.markdown("<div class='header'><div class='title'>CS Genie Pro</div><div class='subtitle'>Chat • Search • Projects • Learn</div></div>", unsafe_allow_html=True)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Chat", "Search", "Projects", "Library", "History"],
    index=0
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "proj_view" not in st.session_state:
    st.session_state.proj_view = {}

# ========================================
# TOP HEADER (visual banner)
# ========================================
st.markdown(
    f"""
    <div class="header">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div>
          <div style="font-size:22px; font-weight:800;">🤖 CS Genie Pro</div>
          <div class="subtitle">Your friendly assistant for computer science concepts</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:14px;" class="muted">Motivation</div>
          <div style="font-weight:700;">{motivation_quote}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================================
# CHAT TAB
# ========================================
if page == "Chat":
    st.markdown("### Chat with CS Genie Pro")
    st.markdown("<div class='card'><small class='muted'>Tip: Ask naturally — e.g., \"What is a stack?\" or \"Explain binary search\"</small></div>", unsafe_allow_html=True)

    # Display chat history with styled bubbles
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    # input
    query = st.chat_input("Ask me anything...") or st.session_state.get("pending_query", "")
    if st.session_state.pending_query:
        st.session_state.pending_query = None

    if query:
        full_query = query
        cleaned = clean_text(full_query)

        # common keywords lists (small, lowercased)
        greetings = ["hi", "hello", "hey", "assalam", "morning", "afternoon", "evening"]
        goodbyes = ["bye", "goodbye", "see you", "tata", "farewell"]
        motivations = ["motivate", "inspire", "encourage", "quote", "motivation"]

        # Save user message
        st.session_state.messages.append({"role": "user", "content": full_query})
        st.markdown(f"<div class='user-bubble'>{full_query}</div>", unsafe_allow_html=True)

        response = None

        # === 1. GREETING (HARDCODED) ===
        if any(g in cleaned.split() for g in greetings):
            response = greeting_response

        # === 2. GOODBYE (HARDCODED) ===
        elif any(g in cleaned for g in goodbyes):
            response = goodbye_response

        # === 3. MOTIVATION (HARDCODED random) ===
        elif any(m in cleaned for m in motivations):
            response = random.choice(motivation_response_list)

        # === 4. EXACT TERM MATCH ===
        elif cleaned in data['cleaned'].values:
            row = data[data['cleaned'] == cleaned].iloc[0]
            term = row['term']
            if row['type'] == 'qa':
                response = f"**Q:** {row['question']}\n\n**A:** {row['answer']}"
            else:
                response = f"**{term.title()}:** {row['definition']}"

        # === 5. TF-IDF SIMILARITY SEARCH ===
        else:
            qvec = vectorizer.transform([cleaned])
            sim = cosine_similarity(qvec, X).flatten()
            idx = sim.argmax()
            best_score = sim[idx]

            if best_score > 0.15:  # Tuned threshold
                row = data.iloc[idx]
                term = row['term']
                if row['type'] == 'qa':
                    response = f"**Q:** {row['question']}\n\n**A:** {row['answer']}\n\n*Matched term: {term.title()}*"
                else:
                    response = f"**{term.title()}:** {row['definition']}\n\n*Similarity: {best_score:.3f}*"
            else:
                # === 6. NO MATCH ===
                suggestions = data[data['type'] == 'term'].sample(min(5, len(data)))
                suggestion_text = "\n".join([f"- **{s['term'].title()}**" for _, s in suggestions.iterrows()])
                response = (
                    "I don't know that yet. Try asking:\n\n"
                    f"{suggestion_text}\n\n"
                    "Examples: *What is a stack?*, *Explain binary search*, *CSS flexbox*"
                )

        # Save & display response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"<div class='bot-bubble'>{response}</div>", unsafe_allow_html=True)

# ========================================
# SEARCH TAB
# ========================================
elif page == "Search":
    st.markdown("### Search CS Terms")
    st.markdown("Search using **natural questions** too!")
    search_term = st.text_input(
        "Enter term or question:",
        placeholder="e.g., what is a stack?, css, algorithm",
        key="search_input"
    )
    search_btn = st.button("Search")

    if search_term and search_btn:
        cleaned = clean_text(search_term)
        qvec = vectorizer.transform([cleaned])
        sim = cosine_similarity(qvec, X).flatten()
        idx = sim.argmax()
        best_score = sim[idx]

        if best_score > 0.15:
            row = data.iloc[idx]
            term = row['term']
            if row['type'] == 'qa':
                st.success(f"**Q:** {row['question']}\n\n**A:** {row['answer']}\n\n*Term: {term.title()}*")
            else:
                st.success(f"**{term.title()}:** {row['definition']}\n\n*Similarity: {best_score:.3f}*")
        else:
            st.warning("No close match found.")
            st.markdown("**Try these:**")
            for _, s in data[data['type'] == 'term'].sample(min(5, len(data))).iterrows():
                st.markdown(f"- **{s['term'].title()}**")

# ========================================
# PROJECTS
# ========================================
elif page == "Projects":
    st.markdown("### Mini Projects for Practice")
    st.markdown("**Click to view code and download!**")

    projects = [
        {
            "name": "Calculator",
            "desc": "GUI calculator with basic operations.",
            "features": ["Add", "Subtract", "Multiply", "Divide"],
            "tech": "Python, Tkinter",
            "code": '''import tkinter as tk\n\ndef calculate():\n    try:\n        result = eval(entry.get())\n        entry.delete(0, tk.END)\n        entry.insert(0, str(result))\n    except:\n        entry.delete(0, tk.END)\n        entry.insert(0, "Error")\n\nroot = tk.Tk()\nroot.title("Calculator")\nroot.geometry("300x400")\n\nentry = tk.Entry(root, font=("Arial", 20), justify="right")\nentry.pack(fill=tk.X, padx=10, pady=10)\n\nbuttons = [('7',1,0),('8',1,1),('9',1,2),('/',1,3),('4',2,0),('5',2,1),('6',2,2),('*',2,3),('1',3,0),('2',3,1),('3',3,2),('-',3,3),('0',4,0),('.',4,1),('=',4,2),('+',4,3)]\nfor (t,r,c) in buttons:\n    cmd = calculate if t == '=' else lambda x=t: entry.insert(tk.END, x)\n    tk.Button(root, text=t, font=("Arial",18), command=cmd).grid(row=r, column=c, padx=5, pady=5, sticky="nsew")\n\ntk.Button(root, text=\"C\", command=lambda: entry.delete(0,tk.END)).grid(row=0, column=0, columnspan=4, sticky=\"nsew\")\nfor i in range(5): root.grid_rowconfigure(i, weight=1); root.grid_columnconfigure(i, weight=1)\nroot.mainloop()''',
            "run_tip": "Run: `python calculator.py`"
        },
        {
            "name": "To-Do List",
            "desc": "Task manager with persistence.",
            "features": ["Add", "Complete", "Delete", "Save"],
            "tech": "Python, JSON, Tkinter",
            "code": '''import tkinter as tk, json, os\nTODO_FILE=\"todo.json\"\ndef load(): return json.load(open(TODO_FILE)) if os.path.exists(TODO_FILE) else []\ndef save(): json.dump(tasks, open(TODO_FILE,\"w\"))\ndef add(): t=entry.get().strip(); if t: tasks.append({\"task\":t,\"done\":False}); save(); update(); entry.delete(0,tk.END)\ndef toggle(i): tasks[i][\"done\"]=not tasks[i][\"done\"]; save(); update()\ndef delete(i): tasks.pop(i); save(); update()\ndef update(): listbox.delete(0,tk.END); [listbox.insert(tk.END, f\"[Completed] {t['task']}\" if t[\"done\"] else f\"[Pending] {t['task']}\") for t in tasks]\nroot=tk.Tk(); root.title(\"To-Do\"); entry=tk.Entry(root,font=14); entry.pack(pady=10,fill=tk.X,padx=20)\ntk.Button(root,text=\"Add\",command=add).pack(pady=5)\nlistbox=tk.Listbox(root,font=12); listbox.pack(fill=tk.BOTH,expand=True,padx=20,pady=10)\ntk.Button(root,text=\"Toggle\",command=lambda: toggle(listbox.curselection()[0]) if listbox.curselection() else None).pack(side=tk.LEFT,expand=True,fill=tk.X,padx=20)\ntk.Button(root,text=\"Delete\",command=lambda: delete(listbox.curselection()[0]) if listbox.curselection() else None).pack(side=tk.RIGHT,expand=True,fill=tk.X,padx=20)\ntasks=load(); update(); root.mainloop()''',
            "run_tip": "Data saved in `todo.json`"
        }
    ]

    for i, proj in enumerate(projects):
        key = proj["name"].lower().replace(" ", "_")
        with st.expander(f"**{proj['name']}** – {proj['desc']}", expanded=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Features:**"); [st.markdown(f"- {f}") for f in proj["features"]]
                st.markdown(f"**Tech:** `{proj['tech']}`")
            with c2:
                view_key = f"view_{key}_{i}"
                st.session_state.proj_view[view_key] = st.button("View Code", key=view_key) or st.session_state.proj_view.get(view_key, False)
                if st.session_state.proj_view[view_key]:
                    st.code(proj["code"], language="python")
                    st.markdown(f"**Run:** {proj['run_tip']}")
                    st.download_button("Download", proj["code"], f"{key}.py", key=f"dl_{i}")

# ========================================
# LIBRARY
# ========================================
elif page == "Library":
    st.markdown("### Learning Resources")
    resources = [
        ("W3Schools", "https://w3schools.com", "HTML, CSS, JS, Python"),
        ("freeCodeCamp", "https://freecodecamp.org", "Full-stack"),
        ("MDN", "https://developer.mozilla.org", "Web standards"),
        ("GeeksforGeeks", "https://geeksforgeeks.org", "DSA, System Design"),
        ("LeetCode", "https://leetcode.com", "Coding practice"),
    ]
    for name, url, desc in resources:
        st.markdown(f"**[{name}]({url})** – {desc}")

# ========================================
# HISTORY
# ========================================
elif page == "History":
    st.markdown("### Chat History")
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            role = "You" if msg["role"] == "user" else "Genie"
            st.write(f"**{role}:** {msg['content']}")
            if st.button("Delete", key=f"del_{i}"):
                st.session_state.messages.pop(i)
                st.experimental_rerun()
        if st.button("Clear All"):
            st.session_state.messages = []
            st.success("Cleared!")
    else:
        st.info("No history yet.")

# ========================================
# FOOTER
# ========================================
st.markdown("""
---
<div class="muted">**CS Genie Pro** – Chat • Search • Projects • Learn &nbsp; • Built with ❤️</div>
""", unsafe_allow_html=True)
