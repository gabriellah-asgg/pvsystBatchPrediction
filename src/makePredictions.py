import pickle
import pandas as pd
# load current best model
best_model = pickle.load(open(r'../res/Canopy_Section_A_BatchResults_all_Panels/SVR_tuned.pkl', 'rb'))

scaler = pickle.load(open(r'../res/Canopy_Section_A_BatchResults_all_Panels/scaler.pkl', 'rb'))

user_input = input("Enter filepath(s) separated by commas to make predictions for data: ")

filepaths = user_input.split(",")
# strip filepaths
filepaths = [f.strip().strip('"') for f in filepaths]

for file in filepaths:
    try:
        df = pd.read_excel(file, names=["Sheds Tilt", "Sheds Azim"])

        # remove all non-numeric data
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(axis=0,inplace=True)

        # standardize data
        scaled_data = scaler.transform(df)

        # make predictions
        predictions = best_model.predict(scaled_data)
        predictions = [round(pred, 4) for pred in predictions]
        df['EArray (KWh)'] = predictions
        output_filepath = file.replace('.xlsx', '_predictions.csv')
        df.to_csv(output_filepath, index=False)
    except FileNotFoundError:
        print("Error file " + file + " could not be found. Please check filepath and try again. \n")
