import pandas as pd
from pycaret.classification import predict_model, load_model

df = pd.read_csv('new_churn_data.csv')

model = load_model('CBC')

predictions = predict_model(model, df)

print (predictions)
