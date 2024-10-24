import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

import predictionGenerator
from sklearn.ensemble import RandomForestRegressor

from preprocessData import Preprocessor
import preprocessData as preprocess
from sklearn.cluster import KMeans

filepath = (
    r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\MOD_Canopy_Batch_Simulation_Project_BatchResults_0.xlsx')
preprocessor = Preprocessor(filepath)
df = preprocessor.read_worksheet(skip_rows=11, columns=["Indent", "Sheds Tilt", "Sheds Azim", "Comment",
                                                        "Error", "Syst_ON", "EArrNom", "GIncLss", "TempLss", "ModQual",
                                                        "MisLoss", "OhmLoss", "EArrMPP", "EArray (KWh)", "EUseful",
                                                        "EffSysR", "EffArrR", "EffArrC", "EffSysC"])



# make a numeric version of the data
numeric_data = df.drop(columns=['Comment', 'Indent', 'Error'])

print(numeric_data.describe())

corr_matrix = numeric_data.corr()
plt.figure()
sns.heatmap(corr_matrix)


# make histograms
for column in numeric_data.columns:
    plt.figure()
    plt.hist(numeric_data[column])
    plt.title(column)

    # create boxplots
    plt.figure()
    sns.boxplot(numeric_data[column])
    plt.title(column)
    plt.autoscale()

# hists show that Syst_ON, MisLoss, EffArrC, and EffSysC are one line and should be dropped


plt.show()

model_df = numeric_data.drop(['Syst_ON', 'MisLoss', 'EffArrC'], axis=1)

# scatterplot
for column in model_df.columns:
    plt.figure()
    plt.scatter(model_df[column], model_df['EArray (KWh)'])
    plt.title(column + " vs EArray (KWh)")
    plt.xlabel(column)
    plt.ylabel('EArray (KWh)')

    plt.figure()
    plt.scatter(model_df['Sheds Tilt'], model_df[column])
    plt.title("Sheds Tilt vs " + column)
    plt.xlabel("Sheds Tilt")
    plt.ylabel(column)

    plt.figure()
    plt.scatter(model_df['Sheds Azim'], model_df[column])
    plt.title("Sheds Azim vs " + column)
    plt.xlabel("Sheds Azim")
    plt.ylabel(column)
plt.show()

model_df = preprocess.process_model_data(numeric_data)
#analyze variable importance

#split and standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)

# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42)

rftree = RandomForestRegressor()
rftree.fit(X_scaled, y)
importances = rftree.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

tree_model = pickle.load(open(r'../res/multitarget_data/DecisionTreeRegressor.pkl', 'rb'))
plt.figure()
tree.plot_tree(tree_model)
plt.show()


