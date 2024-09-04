import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from jsonWriter import *
from modelBuilder import *

rand = 42

filepath_worse = r"Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch Simulation_0815\Semi Opaque Panels_Worst Case Scenario.CSV"
columns = ["Ident", "Sheds Tilt", "Sheds Azim", "NB Strings In Parall", "Comment", "Error", "EArray (KWh)", "LCR Ratio",
           "Ls", "Lc"]
model_df_worst = construct_df(filepath_worse, columns)



# standardize data

model_df_worst = model_df_worst.drop(["LCR Ratio", "Ls", "Lc"], axis=1)

pv_type = filepath_worse.split('\\')[-1].replace(".CSV", "")


filepath = r"Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch Simulation_0815\Canopy_BatchResults_Semi Opaque Panels_Best Case Scenario.xlsx"
model_df_best = construct_df(filepath)

model_df = pd.concat([model_df_worst, model_df_best])
# standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)



# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)


# train model on both best and worst case scenario
model = SVR(C=10, kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# evaluate results
rmse = root_mean_squared_error(y_test, y_pred)
print(rmse)

pickle.dump(model, open(r'../res/Semi Opaque Panels_Worst Case Scenario.CSV/bestAndWorstModel.pkl', 'wb'))
