import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import os
st.write("Files in current directory:", os.listdir())

# Load and prepare data
from pathlib import Path

csv_path = Path(__file__).parent / "melb_data.csv"
df = pd.read_csv(csv_path)

df = df.dropna(axis=0)

y = df.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt',
            'Lattitude', 'Longtitude', 'Propertycount', 'Bedroom2', 'Car', 'Postcode']
X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# Streamlit UI
st.title("🏠 Melbourne House Price Predictor")

st.sidebar.header("Input House Features")
user_input = {feature: st.sidebar.number_input(f"{feature}", value=0.0) for feature in features}
input_df = pd.DataFrame([user_input])

if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")
    st.balloons()  # 🎈 Celebration!

# ✍️ Add credits at the bottom
st.markdown("---")
st.markdown(
    """
    👨‍💻 Built with ❤️ by **Mayan Kumar**  
    📘 [View Source on GitHub](https://github.com/Mayan-kr/melbourne-house-price-predictor)
    """
)
