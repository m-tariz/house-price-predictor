# 🏠 House Price Prediction System

A Machine Learning web app that predicts house prices based on property details such as living area, overall quality, basement size, garage capacity, bedrooms, bathrooms, and year built.

Built as part of the **AIML Summer Internship 2026**, IIHMF, MNNIT Allahabad.

## 🔗 Live App
https://house-price-predictor-72jmreamsukaxkqdizudna.streamlit.app/

## 📁 Project Structure
```
Project_Name/
│
├── Dataset/            # train.csv (Kaggle House Prices dataset)
├── Notebook/            # Google Colab notebook (.ipynb) with EDA, preprocessing, model training
├── Model/               # Saved trained model (.pkl files)
├── Streamlit_App/       # app.py - the deployed prediction web app
├── Documentation/        # Project report
└── README.md
```

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Linear Regression, Random Forest)
- Matplotlib, Seaborn (EDA visualizations)
- Streamlit (deployment)

## ⚙️ How It Works
1. **Data Preprocessing** — handled missing values, removed duplicates, encoded categorical features
2. **EDA** — analyzed distributions and relationships between features and sale price
3. **Feature Engineering** — created new features like `HouseAge` and `TotalBath`
4. **Model Training** — trained and compared Linear Regression and Random Forest models
5. **Evaluation** — assessed models using MAE, RMSE, and R² Score
6. **Deployment** — best performing model deployed as an interactive Streamlit web app

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dataset
[Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## 👤 Author
Mohammad Tariz
