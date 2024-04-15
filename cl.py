import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Check Your Model')
    st.sidebar.markdown('Binnary Web app')

    def load_data():
        data_file = st.sidebar.file_uploader('Upload CSV file')
        data = None
        target_col = None

        if data_file is not None:
            data = pd.read_csv(data_file)
            label = LabelEncoder()
            for col in data.columns:
                data[col] = label.fit_transform(data[col])
            target_col = st.sidebar.selectbox('Select the target column', data.columns)
        return data, target_col

    
    def preprocess_data(df):
        return pd.get_dummies(df)

    def split(df, target_col):
        df = preprocess_data(df)
        x = df.drop(columns=[target_col])
        y = df[target_col]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test, y_pred, class_names):
        if 'Classification Report' in metrics_list:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=class_names)
            st.text(report)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)
       

    df, target_col = load_data()

    if df is not None and target_col:
        x_train, x_test, y_train, y_test = split(df, target_col)
        class_names = df[target_col].unique().astype(str)  # Convert class names to strings

        st.sidebar.subheader("Choose Classifiers")
        classifier = st.sidebar.selectbox("Classifier", ['Logistic Regression'])

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Classification Report', 'Confusion Matrix'))


        if classifier == "Logistic Regression":
            st.sidebar.subheader('Model Hyperparameters')
            C = st.sidebar.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
            max_iter = st.sidebar.slider("Maximum number of iterations", min_value=100, max_value=500, value=200)
            
            if st.sidebar.button("Classify"):
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                accuracy = model.score(x_test, y_test)
                st.subheader(f"Logistic Regression Results")
                st.write(f"Accuracy: {accuracy: .2f}")
                st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
                st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
                plot_metrics(metrics, model, x_test, y_test, y_pred, class_names)

        




    if df is not None and st.sidebar.checkbox("Show raw data", False):
        st.subheader(" Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
