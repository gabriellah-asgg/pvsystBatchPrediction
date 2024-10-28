import pickle
from preprocessData import Preprocessor
import preprocessData as preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

model = pickle.load(
    open(r'..\res\combined_data\MLPRegressor_tuned.pkl', 'rb'))

scaler = pickle.load(
    open(r'../res/combined_data/StandardScaler.pkl', 'rb'))
rand = 42
target = ['EArrNom', 'GIncLss', 'TempLss', 'ModQual', 'OhmLoss', 'EArrMPP',
          'EArray', 'EffSysR', 'EffArrR']
filepath = (
    r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\MOD_Canopy_Batch_Simulation_Project_BatchResults_combined.xlsx')
preprocessor = Preprocessor(filepath)
df = preprocessor.read_worksheet(skip_rows=11,
                                 columns=["Indent", "Sheds Tilt", "Sheds Azim", "Comment", "EArray", "Syst_ON",
                                          "EArrNom", "GIncLss", "TempLss", "ModQual", "OhmLoss", "MisLoss", "EArrMPP",
                                          "EffArrR", "EffSysR", "EffSysC"])
numeric_data = df.drop(columns=['Comment', 'Indent'])
model_df = preprocess.process_model_data(numeric_data)
y = model_df[target]
X = model_df.drop(target, axis=1, inplace=False)
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)
predictions = model.predict(X_test)
rmses = []
for i in range(0, len(target)):
    target_predict = [predict[i] for predict in predictions]
    target_truth = y_test.iloc[:, i].values
    rmse = root_mean_squared_error(target_predict, target_truth)
    rmses.append(rmse)
print(target)
print(rmses)
