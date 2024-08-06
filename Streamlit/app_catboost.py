import streamlit as st
import catboost as cb
import numpy as np
import pickle

# Load default values from the .pkl file
with open('default_values_lat.pkl', 'rb') as file:
    default_values = pickle.load(file)

# Load the trained model
model = cb.CatBoostClassifier()
model.load_model('catboost_model_fifa.cbm', format='cbm')

#st.image('messi_wc.jpeg', caption='Messi with the World Cup', use_column_width=True)


# Custom CSS to set the background image
page_bg_img = f'''
<style>
body {{
background-image: url("https://in.images.search.yahoo.com/search/images;_ylt=AwrKGjFlta5mnX8Jrhy7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3BpdnM-?p=messi+world+cup+wallpaper&fr2=piv-web&type=E210IN105G91674&fr=mcafee#id=1&iurl=https%3A%2F%2Fimg.uhdpaper.com%2Fwallpaper%2Flionel-messi-fifa-world-cup-trophy-2022-winner-960%400%40h-pc-4k.jpg&action=click");
background-size: cover;
}}
</style>
'''

# Inject the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# Streamlit app title
st.title('Football Match Winner Prediction Catboost')

# User input for the features
home_team_name = st.selectbox('Home Team Name', default_values['home_team_names'])
away_team_name = st.selectbox('Away Team Name', default_values['away_team_names'])
home_ppg = st.number_input('Home Team Points per Game', min_value=0.0, max_value=3.0, step=0.01)
away_ppg = st.number_input('Away Team Points per Game', min_value=0.0, max_value=3.0, step=0.01)
away_team_shots_on_target = st.number_input('Away Team Shots on Target', min_value=0, max_value=50, step=1)
home_team_shots_on_target = st.number_input('Home Team Shots on Target', min_value=0, max_value=50, step=1)
stadium_name = st.selectbox('Stadium Name', default_values['stadium_names'])
league_x = st.selectbox('League',default_values['league_names'])
home_team_possession = st.number_input('Home Team Possession (%)', min_value=0, max_value=100, step=1)
away_team_possession = st.number_input('Away Team Possession (%)', min_value=0, max_value=100, step=1)

# When the user clicks the Predict button
if st.button('Predict Winner'):
    # Prepare the feature array
    features = np.array([[home_team_name, away_team_name, home_ppg, away_ppg,
                          away_team_shots_on_target, home_team_shots_on_target,
                          stadium_name, league_x, home_team_possession,
                          away_team_possession]])
    
    # Encode categorical features if necessary
    # Note: Ensure that the categorical encoding matches the one used during model training

    # Predict the winner
    prediction = model.predict(features)

    if prediction == 'home':
        st.success('The predicted result is a Home Team win!')
    elif prediction == 'draw':
        st.success('The predicted result is a Draw!')
    else:
        st.success('The predicted result is an Away Team win!')

# To run the app, save this code in a file named app.py and run the following command:
# streamlit run app.py
