import pickle
import pandas as pd
from preprocessData import *
# load current best model
best_model = pickle.load(open(r'../res/Canopy_BatchResults_Semi Opaque Panels/SVR_tuned.pkl', 'rb'))
scaler = pickle.load(open(r'../res/Canopy_BatchResults_Semi Opaque Panels/StandardScaler.pkl', 'rb'))

user_input = input("Enter filepath(s) separated by commas to make predictions for data: ")

filepaths = user_input.split(",")
# strip filepaths
filepaths = [f.strip().strip('"') for f in filepaths]

for file in filepaths:
    try:
        preprocessor = Preprocessor(file)
        df = preprocessor.read_worksheet(columns=["Sheds Tilt", "Sheds Azim"], sheet="Semi Opaque Orientations ")

        df = process_model_data(df)

        # remove all non-numeric data
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(axis=0,inplace=True)

        # standardize data
        scaled_data = scaler.transform(df)

        # make predictions
        predictions = best_model.predict(scaled_data)
        predictions = [round(pred, 4) for pred in predictions]
        df['EArray (KWh)'] = predictions
        with pd.ExcelWriter(file, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name='SemiOpaquePredictions', index=False)
    except FileNotFoundError:
        print("Error file " + file + " could not be found. Please check filepath and try again. \n")