import pickle
from src.preprocessData import *
# load current best model
best_model = pickle.load(open(r'../res/Canopy_BatchResults_Semi Opaque Panels/SVR_tuned.pkl', 'rb'))
scaler = pickle.load(open(r'../res/Canopy_BatchResults_Semi Opaque Panels/StandardScaler.pkl', 'rb'))

worst_case_model = pickle.load(open(r'../res/Semi Opaque Panels_Worst Case Scenario.CSV/SVR_tuned.pkl', 'rb'))
worst_case_scaler = pickle.load(open(r'../res/Semi Opaque Panels_Worst Case Scenario.CSV/StandardScaler.pkl', 'rb'))


user_input = input("Enter filepath(s) separated by commas to make predictions for data: ")

filepaths = user_input.split(",")
# strip filepaths
filepaths = [f.strip().strip('"') for f in filepaths]

for file in filepaths:
    try:
        preprocessor = Preprocessor(file)
        sheet_name = input("Enter sheet name to read from or hit enter to read from first sheet: ")
        if sheet_name.strip() == "":
            sheet_name = 0
        df = preprocessor.read_worksheet(columns=["Sheds Tilt", "Sheds Azim"], sheet=4)

        df = process_model_data(df)

        # remove all non-numeric data
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(axis=0,inplace=True)

        # standardize data
        scaled_data = scaler.transform(df)

        # make predictions
        predictions = best_model.predict(scaled_data)
        predictions = [round(pred, 4) for pred in predictions]

        # make worst case predictions
        scaled_data_worst_case = worst_case_scaler.transform(df)
        worst_case_predictions = worst_case_model.predict(scaled_data_worst_case)
        worst_case_predictions = [round(pred, 4) for pred in worst_case_predictions]

        df['EArray (KWh)'] = predictions
        df['Worst Case EArray (KWh)'] = worst_case_predictions


        with pd.ExcelWriter(file, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name='SemiOpaquePredictions', index=False)
    except FileNotFoundError:
        print("Error file " + file + " could not be found. Please check filepath and try again. \n")