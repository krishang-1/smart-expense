import joblib

# Load model and vectorizer
model = joblib.load("expense_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("=== Smart Expense Categorizer ===")
print("Type 'exit' to quit.")

while True:
    text = input("\nEnter transaction description: ")
    if text.lower() == 'exit':
        break
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    print(f"Predicted category: {prediction}")
