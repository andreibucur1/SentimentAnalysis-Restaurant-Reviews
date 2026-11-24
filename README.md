# Sentiment Analysis – Restaurant Reviews

This project sorts restaurant reviews into two classes, `1` and `0`, using a **Bag of Words** model with a **Naive Bayes** classifier. Everything is in a single Jupyter notebook, built in Google Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreibucur1/SentimentAnalysis-Restaurant-Reviews/blob/main/costumers_reviews.ipynb)

## Pipeline

```
Restaurant_Reviews.tsv
        │
        ▼
Text cleaning   – keep letters only, lowercase, split into words
                – remove English stopwords (but keep "not")
                – Porter stemming ("loved" → "love", "tasty" → "tasti")
        │
        ▼
Bag of Words    – CountVectorizer, 1,500 most frequent words
        │
        ▼
Train/test split – 80% train / 20% test (random_state = 0)
        │
        ▼
Gaussian Naive Bayes
        │
        ▼
Evaluation      – confusion matrix, accuracy, precision, recall, F1
```

The word **"not"** is kept on purpose. Without it, a review like *"crust not good"* would become *"crust good"* and lose its meaning.

## Results

These are the outputs saved in the notebook. The test set has 200 reviews.

**Confusion matrix** (rows = actual class, columns = predicted class):

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 55 | 42 |
| **Actual 1** | 12 | 91 |

| Metric | Value |
|---|---|
| Accuracy | 0.730 |
| Precision (class 1) | 0.684 |
| Recall (class 1) | 0.883 |
| F1 (class 1) | 0.771 |

In plain terms:
- The model finds **88%** of the reviews that really are class `1`.
- It is too quick to say `1`: **42 of the 97** class-`0` reviews are labeled `1` by mistake. That is why precision is lower than recall.

## Dataset

The dataset is **not included** in this repository. To run the notebook, put a file named `Restaurant_Reviews.tsv` in the same folder. The code expects:

- **Tab-separated** values, with a header row
- A text column named **`Review`**
- The sentiment label (`0` / `1`) in the **last column**
- **At least 1,000 rows.** The cleaning loop is hard-coded to `range(0, 1000)`.

The file is read with `quoting = 3` (`csv.QUOTE_NONE`), so quotation marks inside reviews are treated as normal text.

## How to run

### Google Colab (easiest)

1. Click the **Open in Colab** badge above.
2. In the left sidebar, open **Files** and upload `Restaurant_Reviews.tsv`.
3. Run all cells (**Runtime → Run all**).

### Locally

```bash
git clone https://github.com/andreibucur1/SentimentAnalysis-Restaurant-Reviews.git
cd SentimentAnalysis-Restaurant-Reviews

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib nltk scikit-learn jupyter

# copy Restaurant_Reviews.tsv into this folder, then:
jupyter notebook costumers_reviews.ipynb
```

The notebook calls `nltk.download('stopwords')`, so it needs an internet connection the first time you run it.

## Tech stack

- **Python** (Jupyter / Google Colab)
- **pandas**: loads the dataset
- **NLTK**: stopwords and Porter stemmer
- **scikit-learn**: `CountVectorizer`, `train_test_split`, `GaussianNB`, metrics
- **NumPy**: arrays

## Project structure

```
SentimentAnalysis-Restaurant-Reviews/
├── costumers_reviews.ipynb   # the full pipeline: cleaning → Bag of Words → training → evaluation
└── README.md
```

## Notes

- **Gaussian Naive Bayes** assumes every feature follows a normal distribution. Word counts don't, so treat this model as a baseline.
- The Bag of Words matrix is turned into a dense array (`.toarray()`). That's fine at 1,000 × 1,500, but it uses a lot of memory on larger datasets.
- The trained model and vectorizer are not saved. To classify a new review, run it through the same cleaning steps and the same `cv` object inside the notebook.
- `matplotlib` is imported but not used.
