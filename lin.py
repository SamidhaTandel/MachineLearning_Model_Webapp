import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

def main():
    
    st.title("Linear Regression Web App")
    st.sidebar.title('Linear Regression Web App')

    def load_data():
        data_file = st.sidebar.file_uploader('Upload CSV file')
        if data_file is not None:
            data = pd.read_csv(data_file)
            target_col = st.sidebar.selectbox('Select the target column', data.columns)
            return data, target_col
        return None, None

    def preprocess_data(df):
        df = pd.get_dummies(df)
        return df

    def split(df, target_col):
        # Preprocess data
        df = preprocess_data(df)
        x = df.drop(columns=target_col)
        y = df[target_col]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    df, target_col = load_data()
    if df is not None and target_col:
        x_train, x_test, y_train, y_test = split(df, target_col)

        algorithm = st.sidebar.selectbox("Algorithmn",["Linear Regression"])
        metrics = st.sidebar.multiselect("Metrics to plot", ["MAE", "MSE", "RMSE", "R2 Score"])

        if st.sidebar.button("Run"):
            if algorithm == "Linear Regression":
                # Linear Regression
                model = LinearRegression()
         
            model.fit(x_train, y_train)

     
            y_pred = model.predict(x_test)

            if "MAE" in metrics:
                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"Mean Absolute Error: {mae:.2f}")

            if "MSE" in metrics:
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")

            if "RMSE" in metrics:
                rmse = np.sqrt(mse)
                st.write(f"Root Mean Squared Error: {rmse:.2f}")

            if "R2 Score" in metrics:
                r2 = r2_score(y_test, y_pred)
                st.write(f"R2 Score: {r2:.2f}")

        
        chart = st.sidebar.selectbox("Chart", ["Scatter plot", 'Line Plot'])
        Xaxis = st.sidebar.selectbox("X", df.columns, key='xaxis')
        Yaxis = st.sidebar.selectbox("Y", df.columns, key='yaxis')


        if st.sidebar.button("Plot"):
            if "Scatter plot" in chart:
                plt.figure()
                plt.scatter(df[Xaxis], df[Yaxis])
                plt.xlabel(Xaxis)
                plt.ylabel(Yaxis)
                plt.title("Scatter Plot")
                st.pyplot(plt)

            if "Line Plot" in chart:
                plt.figure()
                plt.plot(df[Xaxis], df[Yaxis])
                plt.xlabel(Xaxis)
                plt.ylabel(Yaxis)
                plt.title("Line Plot")
                st.pyplot(plt)

    if df is not None and st.sidebar.checkbox("Show raw data", False):
        st.subheader("DataSet")
        st.write(df)

if __name__ == '__main__':
    main()
