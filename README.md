# AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator

Module 1 - Data Collection & Pre-Processing 

This module prepares the raw visa dataset so it can be used for machine learning.

- Loaded the dataset  
- Converted date columns into proper datetime format  
- Calculated visa processing time in days  
- Converted text columns into numeric one-hot encoded columns
  
This cleaned the data and made it ready for machine learning


Module 2 - Exploratory Data Analysis (EDA)

This module analyzes visa processing time patterns and extracts meaningful insights from the dataset.

- Analyzed statistical distribution of visa processing time
- Visualized processing time using histogram, boxplot, and scatter plots
- Examined seasonal trends using application month and peak/off-peak categorization
- Studied correlation between application month and processing duration
- Computed country-wise and visa-type-wise average processing times
- Engineered EDA-driven features for downstream modeling

This analysis provided insights into trends, variability, and factors affecting visa processing time

Module 3 - Predictive Modeling

This module builds and evaluates machine learning models to predict visa processing time.

- Trained regression models (Linear Regression(Baseline), Random Forest(Baseline), Random Forest(Tuned))
- Tuned Random Forest hyperparameters using GridSearchCV for optimal performance
- Compared models using evaluation metrics (MAE, RMSE, R² score)
- Examined the results and selected the best model for visa processing dataset.

This provided a reliable model to estimate visa processing durations, facilitating better planning and decision-making.

Module 4 - Web Application Development & Deployment

This module focuses on transforming the trained machine learning model into a usable, end-to-end web application.

- Developed an interactive frontend using Streamlit with a user-friendly input form
- Integrated the backend prediction engine to generate real-time visa processing time estimates
- Implemented input validation and error handling for reliable predictions
- Deployed the application on a cloud platform (Streamlit Community Cloud)
- Performed end-to-end testing using multiple sample scenarios

This enabled users to estimate visa processing time through a fully deployed, production-ready web application.

Deployed URL - https://ai-enabled-visa-status-prediction-and-processing-time-estimator.streamlit.app/
