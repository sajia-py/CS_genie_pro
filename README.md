# CS Genie Pro 🤖

**CS Genie Pro** is an interactive AI-powered assistant designed to help computer science learners **chat, search, and explore CS concepts**. Built with **Streamlit**, it provides instant answers, project examples, and learning resources in a friendly interface.

---

## Features

- **Chat:** Ask questions in natural language and get responses using **TF-IDF similarity search**. Supports greetings, goodbyes, motivational prompts, and CS-related queries.
- **Search:** Quickly find definitions or explanations of CS terms and concepts with similarity matching.
- **Projects:** Explore mini Python projects (e.g., Calculator, To-Do List), view code, and download them for practice.
- **Library:** Access curated resources and tutorials from W3Schools, freeCodeCamp, MDN, GeeksforGeeks, and LeetCode.
- **History:** Keep track of your chat interactions, delete individual messages, or clear all history.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sajia-py/CS_genie_pro.git
cd CS_genie_pro
````

2. **Create a conda environment (recommended):**

```bash
conda create -n cs_genie python=3.10
conda activate cs_genie
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit pandas scikit-learn nltk
```

4. **Download NLTK stopwords** (already handled in `app.py`):

```python
import nltk
nltk.download('stopwords')
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run appp.py
```

* Navigate through **Chat**, **Search**, **Projects**, **Library**, and **History** tabs.
* Ask questions like *“What is a stack?”*, *“Explain binary search”*, or search terms like *“CSS flexbox”*.
* Explore mini projects and download their code for practice.

---

## Project Structure

```
CS_genie_pro/
├─ appp.py             # Main Streamlit app
├─ cs_terms.json       # CS terms dataset
├─ requirements.txt    # Python dependencies (optional)
└─ README.md
```

---

## Technical Highlights

* Built with **Python**, **Streamlit**, **pandas**, **scikit-learn**, and **NLTK**.
* Implements **TF-IDF vectorization** and **cosine similarity** for intelligent query matching.
* Custom CSS for a modern interface with chat bubbles, cards, and sidebar navigation.
* Session state management for chat history and project views.

---

## License

This project is open-source and free to use.
Feel free to contribute or suggest improvements!

---

**CS Genie Pro** – Learn, explore, and practice computer science interactively! 🚀
