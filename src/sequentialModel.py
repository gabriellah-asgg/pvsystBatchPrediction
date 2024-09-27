import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers

from src.modelBuilder import export_model, construct_df

keras.utils.set_random_seed(812)
rand = 42
# construct dataframe for model building
filepath = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\combined_canopy_data.xlsx'
model_df = construct_df(filepath, columns=["Sheds Tilt", "Sheds Azim", "EArray (KWh)"])

# standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)
pv_type = filepath.split('\\')[-1].replace(".xlsx", "")
export_model(scaler, '../res/combined_canopy_data/', False)

# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)

input_layer = tf.input({'shape': [784, ]})
dense1 = tf.layers.dense({'units': 32, 'activation': 'relu'}).apply(input_layer)
dense2 = tf.layers.dense({'units': 64, 'activation': 'sigmoid'}).apply(dense1)
dense3 = tf.layers.dense({'units': 32, 'activation': 'relu'}).apply(dense2)
output_layer = tf.layers.dense({'units': 1}).apply(dense3)

model = tf.model.input({'inputs': input_layer, 'outputs': output_layer})

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.save('my_model')  # Creates a folder named 'my_model'

# Load the model
loaded_model = tf.keras.models.load_model('my_model')