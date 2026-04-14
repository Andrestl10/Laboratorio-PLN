import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

def main():
    mlflow.set_tracking_uri("http://ec2-3-95-197-233.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("baseline_tfidf_lr")
    
    csv_path = r"c:\Users\andre\Documents\Laboratorio-PLN\data\imdb_clean.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check for NaN values in review_clean
    df = df.dropna(subset=['review_clean', 'sentiment'])
    
    X = df['review_clean']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        mlflow.set_tag("username", "johani")
        
        # Define pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score (weighted): {f1}")  
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 5000)
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        print("Model logged to MLflow successfully.")

if __name__ == "__main__":   
    main()
