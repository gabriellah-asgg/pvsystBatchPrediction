import pickle

best_model = pickle.load(
            open(r'../res/Canopy_Section_A_BatchResults_all_Panels/MLPRegressor_tuned.pkl', 'rb'))
print(best_model)