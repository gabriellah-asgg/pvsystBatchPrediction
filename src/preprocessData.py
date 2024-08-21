import pandas as pd


def process_model_data(df_to_process):
    # drop non-numeric columns and columns that are all nan
    model_df = df_to_process.apply(pd.to_numeric, errors='coerce')
    model_df.dropna(axis=1, how='all', inplace=True)

    # drop columns where all values are the same
    cols_to_drop = []
    for col in model_df.columns:
        unique_vals = model_df[col].nunique()
        if unique_vals < 2:
            cols_to_drop.append(col)

    model_df.drop(columns=cols_to_drop, inplace=True)
    return model_df


class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_worksheet(self, skip_rows=0, names=None, index_col=None):
        '''function to preprocess data'''
        display_df = pd.read_excel(self.filepath, skiprows=skip_rows, names=names, index_col=index_col)

        columns = ["Indent", "Sheds Tilt", "Sheds Azim", "NB Strings in Parallel", "NB Inverter or MPPT", "Comment",
                   "Error", "EArray (KWh)"]
        col_dict = {}
        for i, col in enumerate(display_df.columns):
            col_dict[col] = columns[i]
        display_df.rename(columns=col_dict, inplace=True)
        print(display_df.head())
        return display_df
