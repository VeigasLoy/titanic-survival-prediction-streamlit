import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

models = {
    "Random Forest": joblib.load("Random Forest_titanic_model.pkl"),
    "Decision Tree": joblib.load("Decision Tree_titanic_model.pkl"),
    "Logistic Regression": joblib.load("Logistic Regression_titanic_model.pkl"),
}

feature_columns = joblib.load("model_features.pkl")
accuracies_df = pd.read_csv("model_accuracies.csv", index_col=0)

# EDA dataset
eda_data = pd.read_csv("train.csv").dropna(subset=["Survived", "Sex", "Pclass", "Embarked"])

with st.sidebar:
    st.header("Model Selection")
    model_choice = st.radio("Choose a Prediction Model", list(models.keys()))

    st.markdown("---")
    st.markdown("####  Model Descriptions")
    st.markdown("""
    - **Random Forest:** High accuracy, robust to overfitting.  
    - **Decision Tree:** Interpretable, may overfit on small data.  
    - **Logistic Regression:** Simple, effective for binary classification.
    """)

st.title("Titanic Survival Prediction")
st.markdown("""
Fill out the details below to predict your chance of surviving the Titanic disaster.
""")

st.header("Passenger Information")
st.markdown("""
Provide the details of a hypothetical passenger. These inputs are based on real features from Titanic's historical records.
""")

col1, col2 = st.columns(2)

with col1:
    Pclass = st.selectbox(
        "Passenger Class (Socio-economic status)",
        [1, 2, 3],
        help="1 = Upper Class, 2 = Middle Class, 3 = Lower Class"
    )

    Age = st.number_input(
        "Age (in years)",
        min_value=1,
        max_value=80,
        value=25,
        help="Age of the passenger. Younger passengers had varying survival chances."
    )

    Embarked = st.selectbox(
        "Port of Embarkation",
        ["C", "Q", "S"],
        help="C = Cherbourg, Q = Queenstown, S = Southampton"
    )

with col2:
    Sex = st.selectbox(
        "Sex",
        ["male", "female"],
        help="Survival rates differed between men and women."
    )

    SibSp = st.number_input(
        "Siblings/Spouses Aboard",
        0, 10, 0,
        help="Number of siblings or spouses the passenger had aboard the Titanic."
    )

    Parch = st.number_input(
        "Parents/Children Aboard",
        0, 10, 0,
        help="Number of parents or children the passenger had aboard the Titanic."
    )

FamilySize = SibSp + Parch
IsAlone = 1 if FamilySize == 0 else 0
base_fare_dict = {1: 150, 2: 60, 3: 35}
Fare = base_fare_dict[Pclass] * max(1, FamilySize)

st.write(f"**Estimated Fare (auto-calculated):** ${Fare:.2f}")

if st.button("Predict"):
    input_data = {
        "Pclass": Pclass,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "FamilySize": FamilySize,
        "IsAlone": IsAlone,
        "Sex": Sex,
        "Embarked": Embarked,
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"])

    # Add missing cols
    for col in set(feature_columns) - set(input_df.columns):
        input_df[col] = 0
    input_df = input_df[feature_columns]

    model = models[model_choice]
    survival_prob = model.predict_proba(input_df)[0][1]

    # Show prediction
    st.success("Prediction completed!")
    st.subheader("Prediction Result")
    st.metric("Chance of Survival", f"{survival_prob * 100:.2f}%")
    st.progress(int(survival_prob * 100))
    st.caption(f"Model Accuracy: **{accuracies_df.loc[model_choice, 'Accuracy'] * 100:.2f}%**")

# EDA Section
st.markdown("---")
st.header("Dataset Insights")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Survival by Gender")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=eda_data, x="Sex", hue="Survived", ax=ax1)
    ax1.set_title("Survival Count by Gender")
    st.pyplot(fig1)

with col4:
    st.subheader("Survival by Passenger Class")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=eda_data, x="Pclass", hue="Survived", ax=ax2)
    ax2.set_title("Survival Count by Class")
    st.pyplot(fig2)

st.markdown("---")
st.markdown("Made by [Loyston Veigas](https://github.com/VeigasLoy)")
