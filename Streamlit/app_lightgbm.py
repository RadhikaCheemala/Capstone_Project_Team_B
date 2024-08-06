import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle


# Load the trained model and encoders
model = lgb.Booster(model_file='lightgbm_model.txt')
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Custom CSS to set the background image
page_bg_img = '''
<style>
body {
background-image: url("messi_wc.jpg");
background-size: cover;
}
</style>
'''

# Inject the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app title
st.title('Football Match Winner Prediction Lightgbm')

# User input for the features
home_team_name = st.selectbox('Home Team Name', label_encoders['home_team_name'].classes_)
away_team_name = st.selectbox('Away Team Name', label_encoders['away_team_name'].classes_)
home_ppg = st.number_input('Home Team Points per Game', min_value=0.0, max_value=3.0, step=0.01)
away_ppg = st.number_input('Away Team Points per Game', min_value=0.0, max_value=3.0, step=0.01)
away_team_shots_on_target = st.number_input('Away Team Shots on Target', min_value=0, max_value=50, step=1)
home_team_shots_on_target = st.number_input('Home Team Shots on Target', min_value=0, max_value=50, step=1)
stadium_name = st.selectbox('Stadium Name', label_encoders['stadium_name'].classes_)
league_x = st.selectbox('League', label_encoders['league_x'].classes_)
home_team_possession = st.number_input('Home Team Possession (%)', min_value=0, max_value=100, step=1)
away_team_possession = st.number_input('Away Team Possession (%)', min_value=0, max_value=100, step=1)

# When the user clicks the Predict button
if st.button('Predict Winner'):
    input_data = {
        'home_team_name': home_team_name,
        'away_team_name': away_team_name,
        'home_ppg': home_ppg,
        'away_ppg': away_ppg,
        'away_team_shots_on_target': away_team_shots_on_target,
        'home_team_shots_on_target': home_team_shots_on_target,
        'stadium_name': stadium_name,
        'league_x': league_x,
        'home_team_possession': home_team_possession,
        'away_team_possession': away_team_possession
    }

    # Encode the input data
    for col in ['home_team_name', 'away_team_name', 'stadium_name', 'league_x']:
        input_data[col] = label_encoders[col].transform([input_data[col]])[0]

    # Prepare the input data for prediction
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map the prediction back to the original labels
    result_mapping = {0: 'home', 1: 'draw', 2: 'away'}
    result = result_mapping[predicted_class]

    st.success(f'The predicted result is: {result}')
