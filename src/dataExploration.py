import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.preprocessData import Preprocessor
from src import preprocessData as preprocess, predictionGenerator
from sklearn.cluster import KMeans

filepath = (r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch Simulation_0815\Canopy_Section_A_BatchResults_all_Panels.xlsx')
preprocessor = Preprocessor(filepath)
df = preprocessor.read_worksheet(skip_rows=11)


# scatterplot
plt.figure()
plt.scatter(df['Sheds Tilt'], df['EArray (KWh)'])
plt.title("Sheds Tilt vs EArray (KWh)")
plt.xlabel('Sheds Tilt')
plt.ylabel('EArray (KWh)')



plt.figure()
plt.scatter(df['Sheds Azim'], df['EArray (KWh)'])
plt.title("Sheds Azim vs EArray (KWh)")
plt.xlabel('Sheds Azim')
plt.ylabel('EArray (KWh)')

sns.lmplot(x="Sheds Tilt", y="EArray (KWh)", data=df, line_kws={'color': 'yellow'})
plt.title("Sheds Tilt vs EArray (KWh)")
plt.xlabel('Sheds Tilt')
plt.ylabel('EArray (KWh)')

sns.lmplot(x="Sheds Azim", y="EArray (KWh)", data=df, line_kws={'color': 'yellow'})
plt.title("Sheds Azim vs EArray (KWh)")
plt.xlabel('Sheds Azim')
plt.ylabel('EArray (KWh)')

plt.figure()
plt.scatter(df['Sheds Azim'], df['Sheds Tilt'])
plt.title("Sheds Azim vs Sheds Tilt")
plt.xlabel('Sheds Azim')
plt.ylabel('Sheds Tilt')

# make a numeric version of the data
numeric_data = df.drop(columns=['Comment', 'Indent', 'Error'])


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
    plt.boxplot(numeric_data[column])
    plt.title(column)
    plt.autoscale()

# sheds tilt is uni-modal and left-skewed
# sheds azim is bi-modal and mostly centered, slight left-skew
# EArray KWh is one line, so data does not vary much

plt.show()



# make model for k-means clustering
model_df = preprocess.process_model_data(df)
silhouettes = []
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
'''
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    score = silhouette_score(X_train, kmeans.labels_)
    silhouettes.append(score)

# look at elbow plot for best cluster number

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouettes)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
plt.close()
'''

# 2 or 3 is a good number for opaque, 5 for semi opaque
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_train)
clusters = kmeans.predict(X_train)
X_train["Cluster"] = kmeans.labels_


plt.figure(figsize=(8, 6))
plt.scatter(X_train['Sheds Azim'], X_train['Sheds Tilt'], c=X_train['Cluster'], cmap='viridis', s=100, edgecolors='k')
plt.xlabel('Sheds Azim')
plt.ylabel('Sheds Tilt')
plt.title('K-Means Clustering')
plt.colorbar(label='Cluster')
plt.show()



# jointplot
sns.jointplot(x='Sheds Azim', y='Sheds Tilt', data=model_df, kind='scatter')
plt.show()

sns.pairplot(model_df)
plt.show()

# plot actual vs predicted

# load model
#semiopaque code
'''
X_train.drop("Cluster", axis=1, inplace=True)
predictor = predictionGenerator.SemiOpaquePredictor()
predictions_df = predictor.make_predictions(X_test)
worst_case_predictions = predictions_df['Worst Case EArray (KWh)']
best_case_predictions = predictions_df['EArray (KWh)']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['Sheds Azim'], X_test['Sheds Tilt'], y_test, color="black", label="Actual Data")
ax.scatter(X_test['Sheds Azim'], X_test['Sheds Tilt'], worst_case_predictions , color="red", label="SVR Prediction")

ax.set_xlabel('Sheds Azim')
ax.set_ylabel('Sheds Tilt')
ax.set_zlabel('EArray')
plt.title("Worst Case Semi-Opaque SVR Model 3D Plot")
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['Sheds Azim'], X_test['Sheds Tilt'], y_test, color="black", label="Actual Data")
ax.scatter(X_test['Sheds Azim'], X_test['Sheds Tilt'], best_case_predictions, color="red", label="SVR Prediction")

ax.set_xlabel('Sheds Azim')
ax.set_ylabel('Sheds Tilt')
ax.set_zlabel('EArray')
plt.title("Best Case Semi-Opaque SVR Model 3D Plot")
plt.legend()
plt.show()

#plot residuals
residuals = y_test - worst_case_predictions
plt.figure()
plt.scatter(best_case_predictions,residuals, color='green')
plt.axhline(y=0, color='red')
plt.title("Residual Plot for Worst-Case Semi-Opaque Panel Model")
plt.xlabel('Predicted Value')
plt.ylabel('Residuals')
plt.show()
'''

predictor = predictionGenerator.OpaquePredictor()
predictions_df = predictor.make_predictions(X_test)
predictions = predictions_df['EArray (KWh)']

residuals = y_test - predictions
plt.figure()
plt.scatter(predictions,residuals, color='green')
plt.axhline(y=0, color='red')
plt.title("Residual Plot for Opaque Panel Model")
plt.xlabel('Predicted Value')
plt.ylabel('Residuals')
plt.show()