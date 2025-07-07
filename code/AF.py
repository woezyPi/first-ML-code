import streamlit as st
st.set_page_config(page_title="Analyse modèle NLP", layout="wide")

print("----------------Etape 1, Nettoyage des données 💿 ----------------")
import pandas as pd
import numpy as np

#Charge le df et affiche les colonnes et les 5 premières lignes et compte combien de lignes vide
df =pd.read_csv("pr_10K copie.csv", sep=';', encoding='utf-8')
print(df.columns)
print(df.head())
print(df.isnull().sum())

# Supprime les lignes où les colonnes "review" ou "sentiment" sont vides
df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)

# Nettoyage des données textuelles
def clean_text(text):
    text = str(text)   #Convertit en chaîne de caractères
    text = ''.join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()) #Enlève les caractères spéciaux et les met en minuscule
    text = ' '.join(text.split())   #Enlève les espaces multiples
    return text 
# Applique la fonction de nettoyage à la colonne "review"
df["review_clean"] = df["review"].apply(clean_text)

#Enregistre le nouveau df dans un fichier csv clean

print("----------------Etape suivante, evaluation du cleaning des données 📊 ----------------")
df.to_csv("pr_10K_clean.csv", index=False, sep=';', encoding='utf-8')
df =pd.read_csv("pr_10K_clean.csv", sep=';', encoding='utf-8')

print(df.columns)
print(df.head())
print(df.isnull().sum())

lignes_vides = df[df.isnull().any(axis=1)]
print("Index des lignes avec NaN :", lignes_vides.index.tolist())

print("---------------- Étape suivante, vectorisation Bag‑of‑Words + TF‑IDF 🧮 ------------------------")

from sklearn.feature_extraction.text import TfidfVectorizer

# --- TF‑IDF ---
tfidf_vectorizer = TfidfVectorizer(
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b',
    min_df=3
)
X_tfidf = tfidf_vectorizer.fit_transform(df["review_clean"])

print(f"Vocabulaire TF‑IDF : {len(tfidf_vectorizer.get_feature_names_out())} termes")

# Exemple : afficher les 20 premiers mots du vocabulaire TF‑IDF
print("20 premiers tokens (TF‑IDF) :", tfidf_vectorizer.get_feature_names_out()[:20])

tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df.head())

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Re-split rapide (pour la comparaison locale)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])

best_C = 1  # ou grid.best_params_['C'] si déjà calculé
clf = LinearSVC(C=best_C, random_state=42, max_iter=5000)

# TF-IDF
clf.fit(X_train_tfidf, y_train)
acc_tfidf = accuracy_score(y_test, clf.predict(X_test_tfidf))

print(f"Accuracy TF-IDF: {acc_tfidf:.4f}")

print("---------------- Vectorisation terminée ✅ ----------------")

print("---------------- Splitage des donées en 3 💿 ----------------")
from sklearn.model_selection import train_test_split

x = tfidf_df
y = df["sentiment"]
len(x), len(y)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

len(x_train), len(x_val), len(x_test)
print(f"Taille X = {len(x)}, {len(y)}")
print("Taille des ensembles de données :")
print(f"Entraînement : {len(x_train)}")
print(f"Validation : {len(x_val)}")
print(f"Test : {len(x_test)}")
print("Distribution classes (train) :\n", y_train.value_counts(normalize=True))
print("Distribution classes (val)   :\n", y_val.value_counts(normalize=True))
print("Distribution classes (test)  :\n", y_test.value_counts(normalize=True))

total = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
assert total == len(df)

print("---------------- Vérification des données creuses 0️⃣ ----------------")

from scipy import sparse   # déjà dispo via scikit-learn
def check_sparsity(X):
    """Affiche la proportion de zéros dans une matrice creuse X (CSR/CSC)."""
    total = X.shape[0] * X.shape[1]      # nb total de cellules
    nnz   = X.nnz                        # nb de valeurs ≠ 0
    sparsity_pct = 100 * (1 - nnz / total)
    print(f"Shape           : {X.shape}")
    print(f"Non-zéro (nnz)  : {nnz:,}")
    print(f"Sparsité        : {sparsity_pct:.2f} % de zéros")

print("---------------- Test des diffèrents modeles 🦿 ----------------")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

scores = {}
fitted = {}

modelos: dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
    "Multinomial Naive Bayes": MultinomialNB()
}

for name, model in modelos.items():
    print(f"Entraînement du modèle : {name}")
    model.fit(x_train, y_train)
    score = model.score(x_val, y_val)
    scores[name] = score
    fitted[name] = model 
    print(f"Score de validation ({name}) : {score:.4f}")

best_name = max(scores, key=scores.get)
best_model = fitted[best_name]
print (f"Meilleur modèle : {best_name} avec un score de validation de {scores[best_name]:.4f}")

if best_name == "Linear SVM":
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(LinearSVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    print(f"Meilleur paramètre C trouvé via GridSearchCV : {grid.best_params_}")
    print(f"Score cross-validation moyen : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

print("\n=== Récapitulatif des scores ===")
for mod, sc in scores.items():
    print(f"{mod:25s} : {sc:.4f}")

best_model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))

# Évaluation finale
score_test = best_model.score(x_test, y_test)
print(f"🎯 Score final sur le jeu de test : {score_test:.4f}")

print("---------------- Entraînement terminé ✅ ----------------")

print("---------------- Analyse des modèles ecris sur  streamlit avec GPT 4o ----------------")
y_pred = best_model.predict(x_test)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


st.title("📊 Analyse des performances du modèle de sentiment")

# Bloc score final
st.header("🎯 Résultat global")
st.markdown(f"**Modèle sélectionné :** `{best_name}`")
st.markdown(f"**Score sur le jeu de test :** `{score_test:.4f}`")

st.divider()

# Matrice de confusion
st.header("🔍 Matrice de confusion")
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_,
            ax=ax1)
ax1.set_xlabel('Prédit')
ax1.set_ylabel('Réel')
st.pyplot(fig1)

st.divider()

# Rapport classification
st.header("📄 Rapport de classification")
report = classification_report(y_test, y_pred, digits=3)
st.code(report)

st.divider()

# Barplot des modèles
st.header("📊 Comparaison des scores de validation")
fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.barplot(x=list(scores.keys()), y=list(scores.values()), ax=ax2)
ax2.set_ylim(0.7, 1.0)
ax2.set_title("Score de validation (set de validation)")
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Modèle")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
st.pyplot(fig2)

# Analyse des poids si disponible
if hasattr(best_model, "coef_"):
    st.divider()
    st.header("🔎 Mots les plus influents dans le modèle (TF-IDF)")

    coef = best_model.coef_[0]
    vocab = tfidf_vectorizer.get_feature_names_out()
    top_pos_idx = coef.argsort()[-20:][::-1]
    top_neg_idx = coef.argsort()[:20]

    st.subheader("🟩 Positifs")
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    sns.barplot(x=coef[top_pos_idx], y=vocab[top_pos_idx], color='seagreen', ax=ax3)
    ax3.set_title("Top mots positifs")
    st.pyplot(fig3)

    st.subheader("🟥 Négatifs")
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    sns.barplot(x=coef[top_neg_idx], y=vocab[top_neg_idx], color='firebrick', ax=ax4)
    ax4.set_title("Top mots négatifs")
    st.pyplot(fig4)
else:
    st.warning("❗ Ce modèle ne fournit pas de coefficients interprétables.")

st.divider()
st.header("Tester une phrase")

user_input = st.text_input("Test de prédiction des avis :")
if user_input:
    cleaned = clean_text(user_input)
    vector = tfidf_vectorizer.transform([cleaned])
    pred = best_model.predict(vector)[0]
    st.markdown(f"**Sentiment prédit :** `{pred}`")