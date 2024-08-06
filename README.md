# Football Match Prediction

## Overview

This project aims to develop a machine learning model to predict the outcomes of football matches. The predictions categorize the results into three categories: Home win, Away win, or Draw. The model is trained using historical data from various leagues and employs several machine learning algorithms to achieve accurate predictions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Challenges and Limitations](#challenges-and-limitations)
- [Contributors](#contributors)
- [References](#references)

## Dataset

- **Source**: [FootyStats](https://footystats.org/leagues)
- **Coverage**: 2008 to present, including leagues like the English Premier League, Italy Serie A, Spain La Liga, and UEFA Champions League.
- **Size**: 
  - Combined Matches: (21792,67)
  - Combined Players: (58803,278)

## Methodology

1. **Data Collection**: Aggregated data from FootyStats covering matches from 2008-2024.
2. **Data Preprocessing**: Cleaned data, handled missing values, and normalized attributes.
3. **Feature Engineering**: Extracted features like recent team performance and head-to-head stats.
4. **Model Building**: Utilized algorithms such as LightGBM, Random Forest, XGBoost, and CatBoost.
5. **Evaluation**: Assessed models using accuracy, precision, recall, and F1 score metrics.
6. **Deployment**: Developed a dashboard with the top 10 important features for user input.

## Modeling

- **CatBoost**: Effective with categorical features and home/away status.
- **Random Forest**: Less prone to overfitting and ranks feature importance well.
- **LightGBM**: Efficient with large datasets and identifies complex patterns.
- **XGBoost**: Handles missing data well and weighs various factors effectively.

## Evaluation

- **Accuracy by League**:
  - **CatBoost**:
    - Premier League: 68%
    - UEFA Champions League: 67%
    - Serie A: 61%
    - La Liga: 64%
  - **LightGBM**:
    - Premier League: 70%
    - UEFA Champions League: 69%
    - Serie A: 63%
    - La Liga: 65%
  - **XGBoost**:
    - Premier League: 67%
    - UEFA Champions League: 67%
    - Serie A: 63%
    - La Liga: 66%
  - **Random Forest**:
    - Premier League: 64%
    - UEFA Champions League: 65%
    - Serie A: 57%
    - La Liga: 57%


## Challenges and Limitations

- **Data Quality and Availability**: Inconsistent or missing player statistics and limited historical data for newer teams or players.
- **Feature Selection**: Balancing the number of features for optimal performance.
- **Evaluation Metrics**: Addressing class imbalance and selecting appropriate metrics.

## Contributors

- Radhika Cheemala
- Anurag Kunchi
- Suhana Shaik

## References

- [FootyStats](https://footystats.org/leagues)
- [FIFA 23 Ratings Hub - EA SPORTS Official Site](https://www.ea.com/fifa/ratings)
- [Premier League Football News, Fixtures, Scores & Results](https://www.premierleague.com/)
