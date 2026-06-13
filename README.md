# Information Retrieval System — BBC News
## BITS Pilani | AIMLCZG537/DSECLZG537 | Assignment 1 | S2-25

---

## 📁 Project Structure

```
ir_assignment/
├── app.py                  ← Main Streamlit application
├── ir_data/
│   └── bbc_news.csv        ← BBC News dataset (20 articles, 6 categories)
├── README.md               ← This file
└── requirements.txt        ← Python dependencies
```

---

## ⚙️ Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.0.0
```

> **Note:** The app uses **zero external NLP libraries** (no NLTK, no spaCy).  
> All tokenization, stemming (Porter), lemmatization, and phonetic algorithms  
> are implemented from scratch in pure Python.

---

## 🚀 Installation & Running

### Step 1 — Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Using the app
1. Open the browser at `http://localhost:8501`
2. Navigate to **📁 Upload & View Dataset**
3. Click **"Load Built-in BBC Dataset"** (or upload your own CSV)
4. Click **"Build All Indexes"**
5. Explore all features using the sidebar navigation

---

## 📋 Features Implemented

| Page | Feature |
|------|---------|
| Upload & View | CSV upload, category distribution chart, document viewer |
| Text Preprocessing | Tokenization, lowercasing, stop word removal, hyphen handling, stemming, lemmatization, inverted index lookup, vocab comparison |
| Phrase Query | Biword index, positional index, false positive detection, index visualisation |
| Dictionary Search | BST, B-Tree (order 4), single/batch experiments, performance charts |
| Tolerant Retrieval | Wildcard + K-gram, spelling correction, edit distance, Soundex phonetic |
| Inference | Full discussion answering all 7 required questions, rubric self-assessment |

---

## 📂 Dataset Format

The CSV must have these columns:
- `id` — unique document identifier (e.g. bbc_001)
- `category` — article category (technology, politics, sport, etc.)
- `title` — article headline
- `text` — full article body

---

## 🖥️ BITS Virtual Lab

To run on the BITS Lab portal:
1. Upload the entire `ir_assignment/` folder
2. Install dependencies: `pip install streamlit pandas numpy scikit-learn`
3. Run: `streamlit run app.py`
4. Take screenshots of every page for your report

---

## 📊 Assignment Coverage

- ✅ Streamlit end-to-end workflow (1 mark)
- ✅ Text preprocessing — all 5 steps (1.5 marks)
- ✅ Stemming vs Lemmatization comparison (1 mark)
- ✅ Biword + Positional index phrase query (1.5 marks)
- ✅ BST + B-Tree dictionary search (1.5 marks)
- ✅ Tolerant retrieval — 4 methods (1.5 marks)
- ✅ Experimental evidence & inferences (1 mark)
- ⬜ Virtual lab usage — run & screenshot (1 mark)

**Total possible: 10/10**
