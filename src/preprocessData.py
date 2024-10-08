import pandas as pd


def process_model_data(df_to_process):
    input_features = ['Sheds Tilt', 'Sheds Azim']
    # drop non-numeric columns and columns that are all nan
    model_df = df_to_process.apply(pd.to_numeric, errors='coerce')
    model_df.dropna(axis=1, how='all', inplace=True)

    #drop rows that are all nan
    model_df.dropna(axis=0, how='all', inplace=True)

    # drop columns where all values are the same
    cols_to_drop = []
    for col in model_df.columns:
        unique_vals = model_df[col].nunique()
        if unique_vals < 2 and col not in input_features:
            cols_to_drop.append(col)

    model_df.drop(columns=cols_to_drop, inplace=True)
    model_df.dropna(axis=0, inplace=True)
    return model_df


def remove_non_numeric(df):
    # remove all non-numeric data
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(axis=0, inplace=True)
    return df


class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_worksheet(self, skip_rows=0, names=None, index_col=None, columns=None, sheet=0):
        '''function to preprocess data'''
        if columns is None:
            columns = ["Indent", "Sheds Tilt", "Sheds Azim", "NB Strings in Parallel", "NB Inverter or MPPT", "Comment",
                       "Error", "EArray (KWh)"]
        display_df = pd.read_excel(self.filepath, skiprows=skip_rows, names=names, index_col=index_col,
                                   sheet_name=sheet)
        display_df = display_df.dropna(axis=1, how="all")
        col_dict = {}
        for i, col in enumerate(display_df.columns):
            col_dict[col] = columns[i]
        display_df.rename(columns=col_dict, inplace=True)
        print(display_df.head())
        return display_df

    def select_worksheet(self, skip_rows=0, names=None, index_col=None, column_names=None):
        if column_names is None:
            column_names = ["Indent", "Sheds Tilt", "Sheds Azim", "NB Strings in Parallel", "NB Inverter or MPPT",
                            "Comment",
                            "Error", "EArray (KWh)"]
        display_df = pd.read_excel(self.filepath, skiprows=skip_rows, names=names, index_col=index_col,
                                   sheet_name=None)
        for sheet_name in display_df.items():
            temp_df = display_df[sheet_name]
            new_sheet = sheet_name.strip()
        col_dict = {}
        for i, col in enumerate(display_df.columns):
            col_dict[col] = column_names[i]
        display_df.rename(columns=col_dict, inplace=True)
        print(display_df.head())
        return display_df

    def read_csv(self, skip_rows=0, names=None, index_col=None, columns=None):
        '''function to preprocess data'''
        if columns is None:
            columns = ["Indent", "Sheds Tilt", "Sheds Azim", "NB Strings in Parallel", "NB Inverter or MPPT", "Comment",
                       "Error", "EArray (KWh)"]
        display_df = pd.read_csv(self.filepath, skiprows=skip_rows, names=names, index_col=index_col)
        col_dict = {}
        for i, col in enumerate(display_df.columns):
            col_dict[col] = columns[i]
        display_df.rename(columns=col_dict, inplace=True)
        print(display_df.head())
        return display_df
