import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
import gensim.downloader as api

def main():
    mlflow.set_tracking_uri("http://ec2-3-95-197-233.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("word2vec_pretrained_embeddings")

    csv_path = "/home/johan/Laboratorio-PLN/data/imdb_clean.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['review_clean', 'sentiment'])

    X = df['review_clean'].astype(str).values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ─── Hiperparámetros (mismos que paso 3) ────────────────────────────────
    MAX_TOKENS    = 20000
    SEQUENCE_LEN  = 200
    EMBEDDING_DIM = 300   # Word2Vec google-news usa 300 dimensiones
    HIDDEN_UNITS  = [128, 64]
    DROPOUT_RATE  = 0.4
    EPOCHS        = 10
    BATCH_SIZE    = 64
    # ────────────────────────────────────────────────────────────────────────

    # TextVectorization (requerido por el enunciado para redes neuronales)
    vectorizer = keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode='int',
        output_sequence_length=SEQUENCE_LEN
    )
    vectorizer.adapt(X_train)

    # Convertir texto a secuencias de enteros
    X_train_seq = vectorizer(X_train).numpy()
    X_test_seq  = vectorizer(X_test).numpy()

    # ── Cargar Word2Vec y construir matriz de embeddings ─────────────────────
    print("Cargando Word2Vec (google-news-300)... esto puede tardar unos minutos.")
    w2v_model = api.load("word2vec-google-news-300")
    print("Word2Vec cargado exitosamente.")

    vocab = vectorizer.get_vocabulary()
    embedding_matrix = np.zeros((MAX_TOKENS, EMBEDDING_DIM))
    found = 0
    for idx, word in enumerate(vocab):
        if idx >= MAX_TOKENS:
            break
        if word in w2v_model:
            embedding_matrix[idx] = w2v_model[word]
            found += 1
    print(f"Palabras encontradas en Word2Vec: {found}/{min(len(vocab), MAX_TOKENS)}")

    # Construcción del modelo con embeddings CONGELADOS
    model = keras.Sequential([
        keras.layers.Embedding(
            input_dim=MAX_TOKENS,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=SEQUENCE_LEN,
            trainable=False,           # ← CONGELADO
            name="word2vec_embedding_frozen"
        ),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(HIDDEN_UNITS[0], activation='relu'),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(HIDDEN_UNITS[1], activation='relu'),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    with mlflow.start_run(run_name="word2vec_frozen"):
        mlflow.set_tag("username", "johani")
        mlflow.set_tag("step", "4_word2vec_frozen")
        mlflow.set_tag("embedding_type", "Word2Vec")
        mlflow.set_tag("trainable", "False")

        # Log parámetros
        mlflow.log_param("max_tokens", MAX_TOKENS)
        mlflow.log_param("sequence_len", SEQUENCE_LEN)
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("hidden_units", str(HIDDEN_UNITS))
        mlflow.log_param("dropout_rate", DROPOUT_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("embedding_source", "word2vec-google-news-300")
        mlflow.log_param("embeddings_trainable", False)

        # Entrenamiento
        print("Training model...")
        history = model.fit(
            X_train_seq, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1
        )

        # Evaluación
        print("Evaluating model...")
        y_prob = model.predict(X_test_seq)
        y_pred = (y_prob >= 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # Log métricas finales
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # Log métricas por época
        for epoch, (loss, val_loss, acc_e, val_acc) in enumerate(zip(
            history.history['loss'],
            history.history['val_loss'],
            history.history['accuracy'],
            history.history['val_accuracy']
        )):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", acc_e, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Log modelo
        mlflow.tensorflow.log_model(model, "model")
        print("Model logged to MLflow successfully.")

if __name__ == "__main__":
    main()
