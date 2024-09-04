from abc import ABC, abstractmethod
import pickle


class Predictor(ABC):
    def __init__(self):
        self.best_model = None
        self.scaler = None
        self.load_model_and_scaler()

    @abstractmethod
    def make_predictions(self, df):
        pass

    @abstractmethod
    def load_model_and_scaler(self):
        pass


class OpaquePredictor(Predictor):

    def make_predictions(self, df):
        scaled_data = self.scaler.transform(df)
        predictions = self.best_model.predict(scaled_data)
        predictions = [round(pred, 4) for pred in predictions]
        df['EArray (KWh)'] = predictions
        return df

    def load_model_and_scaler(self):
        self.best_model = pickle.load(
            open(r'../res/Canopy_Section_A_BatchResults_all_Panels/SVR_tuned.pkl', 'rb'))

        self.scaler = pickle.load(open(r'../res/Canopy_Section_A_BatchResults_all_Panels/StandardScaler.pkl', 'rb'))


class SemiOpaquePredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.worst_case_scaler = None
        self.worst_case_model = None
        self.load_worst_case_model_and_scaler()

    def load_model_and_scaler(self):
        self.best_model = pickle.load(
            open(r'../res/Canopy_BatchResults_Semi Opaque Panels_Best Case Scenario/SVR_tuned.pkl', 'rb'))
        self.scaler = pickle.load(open(r'../res/Canopy_BatchResults_Semi Opaque Panels_Best Case Scenario/StandardScaler.pkl', 'rb'))

    def load_worst_case_model_and_scaler(self):
        self.worst_case_model = pickle.load(
            open(r'../res/Semi Opaque Panels_Worst Case Scenario.CSV/SVR_tuned.pkl', 'rb'))
        self.worst_case_scaler = pickle.load(
            open(r'../res/Semi Opaque Panels_Worst Case Scenario.CSV/StandardScaler.pkl', 'rb'))

    def make_predictions(self, df):
        scaled_data = self.scaler.transform(df)
        predictions = self.best_model.predict(scaled_data)
        predictions = [round(pred, 4) for pred in predictions]

        # make worst case predictions
        scaled_data_worst_case = self.worst_case_scaler.transform(df)
        worst_case_predictions = self.worst_case_model.predict(scaled_data_worst_case)
        worst_case_predictions = [round(pred, 4) for pred in worst_case_predictions]

        df['EArray (KWh)'] = predictions
        df['Worst Case EArray (KWh)'] = worst_case_predictions

        return df
