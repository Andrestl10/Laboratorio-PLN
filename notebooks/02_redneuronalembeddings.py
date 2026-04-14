import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras

def main():
    mlflow.set_tracking_uri("http://ec2-3-95-197-233.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("baseline_tfidf_lr")

    csv_path = "/home/johan/Laboratorio-PLN/data/imdb_clean.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['review_clean', 'sentiment'])

    X = df['review_clean'].astype(str).values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ─── Hiperparámetros justificados ───────────────────────────────────────
    MAX_TOKENS    = 20000   # vocabulario suficiente para reseñas de cine
    SEQUENCE_LEN  = 200     # reseñas IMDB suelen tener ~200-300 tokens relevantes
    EMBEDDING_DIM = 64      # balance entre capacidad representacional y costo
    HIDDEN_UNITS  = [128, 64]  # dos capas: extrae patrones complejos sin overfitting
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

    # Construcción del modelo
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        keras.layers.Embedding(
            input_dim=MAX_TOKENS,
            output_dim=EMBEDDING_DIM,
            name="trainable_embedding"
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

    with mlflow.start_run(run_name="red_neuronal_embeddings_entrenables"):
        mlflow.set_tag("username", "johan")
        mlflow.set_tag("step", "3_entrenamiento_redneuronal-embeddings")

        # Log parámetros
        mlflow.log_param("max_tokens", MAX_TOKENS)
        mlflow.log_param("sequence_len", SEQUENCE_LEN)
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("hidden_units", str(HIDDEN_UNITS))
        mlflow.log_param("dropout_rate", DROPOUT_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("embeddings_trainable", True)

        # Entrenamiento
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1
        )

        # Evaluación
        print("Evaluating model...")
        y_prob = model.predict(X_test)
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