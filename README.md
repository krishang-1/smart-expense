# Smart Expense Categorizer

A simple AI-powered tool to categorize your daily expenses automatically.  
It uses machine learning (TF-IDF + Logistic Regression) to classify transactions into categories such as Food, Travel, Shopping, Utilities, and more.

---

## Features

- AI agent for categorizing transactions from SMS or bank statements.
- Supports categories like Food, Travel, Shopping, Utilities, Health, Education, Entertainment, and Others.
- Easy-to-use Python scripts for training and categorizing new transactions.
- Lightweight and runs on local machine using scikit-learn.

---

## Dataset

A sample dataset (`transactions.csv`) with labeled transactions is included:

| Transaction          | Category      |
|---------------------|---------------|
| Swiggy 250          | Food          |
| Uber 300            | Travel        |
| Big Bazaar 500      | Grocery       |
| Zomato 350          | Food          |
| Amazon 1200         | Shopping      |
| ...                 | ...           |

> 30–40 entries included for demonstration. You can add more transactions as needed.

---

## Requirements

- Python 3.9+
- pandas
- scikit-learn

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python train.py
```

This will train the Logistic Regression model using the dataset and save the model and vectorizer.

### 2. Categorize Transactions

```bash
python categorize.py
```

Enter transaction descriptions to get AI-suggested categories. Example:

```
Enter transaction: Swiggy 250
Predicted Category: Food
```

Type `exit` to quit.

---

## Project Structure

```
smart-expense/
│
├─ train.py           # Script to train the ML model
├─ categorize.py      # Script to categorize new transactions
├─ transactions.csv   # Sample dataset
├─ model.pkl          # Saved trained model
├─ vectorizer.pkl     # Saved TF-IDF vectorizer
└─ README.md
```

---

## Contributing

Contributions are welcome! You can:

- Add more categories
- Improve ML accuracy with more data or advanced models
- Implement a GUI or web interface

---

## License

This project is open-source and free to use.
