import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import io
import base64
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")

df = pd.read_csv('loan_data.csv')

df.dropna(inplace=True)
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})
df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(int)
df['Property_Area'] = df['Property_Area'].map({'Urban': 0, 'Semiurban': 1, 'Rural': 2})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})


X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'loan_status_model_gb.pkl')

model = pipeline.named_steps['classifier']
feature_importance = pd.Series(model.feature_importances_, index=X.columns)

pipeline = joblib.load('loan_status_model_gb.pkl')

st.title("Kreditbewilligungs-System")

st.markdown("""
Dies ist ein Kreditbewilligungs-System. Sie können in der Sidebar die Daten eingeben und prüfen, ob Ihr Kredit bewilligt werden würde. Außerdem finden Sie nützliche Visualisierungen, um herauszufinden, welche Kreditkonditionen wichtig für die Vergabe sind, anhand dieses Modells.
""")

st.sidebar.header("Eingabewerte")
gender = st.sidebar.selectbox("Geschlecht", ["Male", "Female"])
married = st.sidebar.selectbox("Verheiratet", ["Yes", "No"])
coapplicant = st.sidebar.selectbox("Ist ein Mitantragsteller vorhanden?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Anzahl der Angehörigen", [0, 1, 2, 3, 4])
education = st.sidebar.selectbox("Bildungsstatus", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Selbstständig", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Einkommen des Antragstellers", min_value=0)
coapplicant_income = st.sidebar.number_input("Einkommen des Mitantragstellers", min_value=0) if coapplicant == "Yes" else 0
loan_amount = st.sidebar.number_input("Kreditsumme", min_value=0)
loan_amount_term = st.sidebar.number_input("Laufzeit des Kredits in Monaten", min_value=0)
credit_history = st.sidebar.selectbox("Kreditgeschichte (0: Schlechte, 1: Gute)", [0, 1])
property_area = st.sidebar.selectbox("Gebiet der Immobilie", ["Urban", "Semiurban", "Rural"])

input_data = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],
    'Married': [1 if married == "Yes" else 0],
    'Dependents': [dependents],
    'Education': [1 if education == "Graduate" else 0],
    'Self_Employed': [1 if self_employed == "Yes" else 0],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [0 if property_area == "Urban" else 1 if property_area == "Semiurban" else 2]
})

def create_histplot(data, title, color):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    sns.histplot(data, kde=True, ax=ax, color=color)
    ax.set_title(title, color='white')
    ax.set_xlabel(title, color='white')
    ax.set_ylabel('Häufigkeit', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.tick_params(colors='white')
    return fig

fig1 = create_histplot(df['ApplicantIncome'], "Einkommen des Antragstellers", "skyblue")
fig2 = create_histplot(df['CoapplicantIncome'], "Einkommen des Mitantragstellers", "green")
fig3 = create_histplot(df['LoanAmount'], "Kreditsumme", "orange")
fig4 = create_histplot(df['Loan_Amount_Term'], "Laufzeit des Kredits (Monate)", "purple")
fig5 = create_histplot(df['Credit_History'], "Kreditgeschichte", "red")

st.sidebar.title("Visualisierungen")
visualization = st.sidebar.selectbox("Wählen Sie eine Visualisierung aus:", [
    "Keine",
    "Verteilung des Antragsteller-Einkommens",
    "Kreditsumme",
    "Kreditgeschichte",
    "Kreditsumme nach Einkommen",
    "Immobiliengebiet",
    "Bildungsstatus"
])

def plot_applicant_income_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df, x='ApplicantIncome', hue='Loan_Status', kde=True, ax=ax, palette="viridis", multiple="stack")
    ax.set_title("Verteilung des Antragsteller-Einkommens", color='white')
    ax.set_xlabel('Einkommen', color='white')
    ax.set_ylabel('Häufigkeit', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)

def plot_loan_amount_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df, x='LoanAmount', hue='Loan_Status', kde=True, ax=ax, palette="viridis", multiple="stack")
    ax.set_title("Kreditsumme", color='white')
    ax.set_xlabel('Betrag', color='white')
    ax.set_ylabel('Häufigkeit', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)

def plot_credit_history_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df, x='Credit_History', hue='Loan_Status', kde=False, ax=ax, palette="viridis", multiple="stack")
    ax.set_title("Kreditgeschichte", color='white')
    ax.set_xlabel('Kreditgeschichte', color='white')
    ax.set_ylabel('Häufigkeit', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)

def plot_loan_amount_vs_income():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df, ax=ax, palette="viridis")
    ax.set_title("Kreditsumme nach Einkommen", color='white')
    ax.set_xlabel('Einkommen', color='white')
    ax.set_ylabel('Kreditsumme', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)

def plot_property_area_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Property_Area', hue='Loan_Status', data=df, ax=ax, palette="viridis")
    ax.set_title("Immobiliengebiet", color='white')
    ax.set_xlabel('Gebiet', color='white')
    ax.set_ylabel('Häufigkeit', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)

def plot_education_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Education', hue='Loan_Status', data=df, ax=ax, palette="viridis")
    ax.set_title("Bildungsstatus", color='white')
    ax.set_xlabel('Bildungsstatus', color='white')
    ax.set_ylabel('Häufigkeit', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)

if visualization == "Verteilung des Antragsteller-Einkommens":
    plot_applicant_income_distribution()
elif visualization == "Kreditsumme":
    plot_loan_amount_distribution()
elif visualization == "Kreditgeschichte":
    plot_credit_history_distribution()
elif visualization == "Kreditsumme nach Einkommen":
    plot_loan_amount_vs_income()
elif visualization == "Immobiliengebiet":
    plot_property_area_distribution()
elif visualization == "Bildungsstatus":
    plot_education_distribution()

if st.sidebar.button("Kreditbewilligung prüfen"):
    prediction = pipeline.predict(input_data)
    st.write(f"Vorhersagewert: {prediction[0]}")  
    if prediction[0] == 1:
        st.success("Der Kredit wird bewilligt.")
    else:
        st.error("Der Kredit wird nicht bewilligt.")
    
    st.subheader("Merkmalswichtigkeit")
    feature_importance = pd.Series(pipeline.named_steps['classifier'].feature_importances_, index=input_data.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
    ax.set_title('Merkmalswichtigkeit', color='white')
    ax.set_xlabel('Wichtigkeit', color='white')
    ax.set_ylabel('Merkmal', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='white')
    st.pyplot(fig)
    st.write("Die obige Grafik zeigt die Wichtigkeit der einzelnen Merkmale für die Kreditentscheidung.")
