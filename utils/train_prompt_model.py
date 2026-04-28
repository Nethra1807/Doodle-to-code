import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_and_save():
    # Sample data
    data = [
        ("Login page", "form"),
        ("Create a place for users to sign in", "form"),
        ("User profile with avatar", "profile"),
        ("Dashboard with sidebar", "dashboard"),
        ("E-commerce homepage", "ecommerce"),
        ("Registration screen", "form"),
        ("Analytics dashboard with charts", "dashboard"),
        ("Online store showing products", "ecommerce"),
        ("A profile settings page", "profile"),
        ("Sign up form", "form"),
        ("Admin dashboard", "dashboard"),
        ("Shopping cart", "ecommerce"),
        ("User account details", "profile"),
        ("Contact us page with message box", "form"),
        ("Product listing page", "ecommerce"),
        ("Forgot password screen", "form"),
        ("Metrics overview", "dashboard"),
        ("Store checkout page", "ecommerce"),
        ("Edit profile information", "profile"),
        ("Pricing table for subscription plans", "table"),
        ("Data table with sortable columns", "table"),
        ("User management table", "table"),
        ("Product inventory list", "table"),
        ("Contact list with search", "list"),
        ("Hero section for a startup", "landing"),
        ("Landing page with features and CTA", "landing"),
        ("Navigation bar with links", "navbar"),
        ("Site header with logo and menu", "navbar"),
        ("Mobile responsive navbar", "navbar"),
        ("Top menu bar for navigation", "navbar")
    ]
    
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    model = MultinomialNB()
    model.fit(X, labels)
    
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
        
    print(f"✅ Saved prompt models (model.pkl and vectorizer.pkl) to {model_dir}")

if __name__ == "__main__":
    train_and_save()
