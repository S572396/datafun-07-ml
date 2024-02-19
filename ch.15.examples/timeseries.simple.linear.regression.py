import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Read the data
nyc = pd.read_csv('ave_hi_nyc_jan_1895-2018.csv')
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc['Date'] = nyc['Date'] // 100  # Using floor division instead of floordiv
print(nyc.head(3))

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(nyc[['Date']], nyc['Temperature'])

# Display the coefficients and intercept
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Make predictions on a range of years using np.arange
x_range = np.arange(min(nyc['Date']), max(nyc['Date']) + 1).reshape(-1, 1)
y_pred_range = model.predict(x_range)

# Display the predicted values
print("Predicted Temperatures:", y_pred_range)

# Predict temperatures for specific years
def predict_temperature(year):
    return model.coef_[0] * year + model.intercept_


# Predict temperatures for years:
years_to_predict = [1890, 2019, 2024]
predicted_temperatures = [predict_temperature(year) for year in years_to_predict]


for year, temperature in zip(years_to_predict, predicted_temperatures):
    print(f"Predicted Temperature in {year}: {temperature:.2f}")


plt.scatter(nyc['Date'], nyc['Temperature'], color='blue', label='Actual Temperatures')

# regression line
plt.plot(x_range, y_pred_range, color='red', linewidth=2, label='Linear Regression')


plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()
