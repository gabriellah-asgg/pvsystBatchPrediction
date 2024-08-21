import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessData import Preprocessor
import preprocessData

filepath = (r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch '
            r'Simulation_0815\Canopy_Section_A_BatchResults_all_Panels.xlsx')
preprocessor = Preprocessor(filepath)
df = preprocessor.read_worksheet(skip_rows=11)

# scatterplot
plt.figure()
plt.scatter(df['Sheds Tilt'], df['EArray (KWh)'])


plt.figure()
plt.scatter(df['Sheds Azim'], df['EArray (KWh)'])

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

# sheds tilt is uni-modal and left-skewed
# sheds azim is bi-modal and mostly centered, slight left-skew
# EArray KWh is one line, so data does not vary much

plt.show()