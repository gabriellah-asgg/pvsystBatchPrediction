import pickle
import tkinter
from tkinter import filedialog
import os
import pandas as pd
from src.preprocessData import Preprocessor, process_model_data


def preprocess_sheet(preprocess, scale, curr_sheet):
    df = preprocess.read_worksheet(columns=["Sheds Tilt", "Sheds Azim"], sheet=curr_sheet)

    df = process_model_data(df)

    # remove all non-numeric data
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(axis=0, inplace=True)

    # standardize data
    scaled_data = scale.transform(df)

    return scaled_data, df


class Gui():
    def __init__(self):
        self.selected_radio = None
        self.sheet_names_to_process = []
        self.best_model = None
        self.scaler = None

    def click_submit_radio(self, radio_win):
        selected = self.selected_radio.get()
        if selected == "Opaque Panels":
            print("You selected Opaque Panels.")
            # load current best model

            self.best_model = pickle.load(
                open(r'../res/Canopy_Section_A_BatchResults_all_Panels/SVR_tuned.pkl', 'rb'))

            self.scaler = pickle.load(open(r'../res/Canopy_Section_A_BatchResults_all_Panels/StandardScaler.pkl', 'rb'))
        elif selected == "Semi-Opaque Panels":
            print("You selected Semi-Opaque Panels.")
            self.best_model = pickle.load(
                open(r'../res/Canopy_Section_A_BatchResults_all_Panels/SVR_tuned.pkl', 'rb'))

            self.scaler = pickle.load(open(r'../res/Canopy_Section_A_BatchResults_all_Panels/StandardScaler.pkl', 'rb'))
        radio_win.destroy()  # Close the window

    def click_submit_listbox(self, sheet_listbox, sheet_listbox_win):
        selected_indices = sheet_listbox.curselection()
        selected_items = [sheet_listbox.get(i) for i in selected_indices]
        print("Selected items:", selected_items)
        self.sheet_names_to_process = selected_items
        sheet_listbox_win.destroy()

    def make_listbox(self, sheet_names_list):
        listbox_win = tkinter.Tk()
        listbox_win.title("Select a sheet(s) to add predictions for:")
        listbox = tkinter.Listbox(listbox_win, height=20, width=50, selectmode='multiple')

        for i, sheet in enumerate(sheet_names_list.keys()):
            listbox.insert(i, sheet)

        listbox.pack(pady=10)
        submit_button = tkinter.Button(listbox_win, text='Submit',
                                       command=lambda: self.click_submit_listbox(listbox, listbox_win))
        submit_button.pack(pady=5)

        listbox_win.geometry("600x400")
        listbox_win.mainloop()

    def run(self):

        files = tkinter.filedialog.askopenfilenames(title="Select file to read from:")
        print(files)

        radio_win = tkinter.Tk()
        self.selected_radio = tkinter.StringVar()
        self.selected_radio.set(' ')
        radio_win.title("Select Panel Prediction Type:")

        radio_opaque = tkinter.Radiobutton(radio_win, text="Opaque Panels", variable=self.selected_radio,
                                           value="Opaque Panels")
        radio_semiopaque = tkinter.Radiobutton(radio_win, text="Semi-Opaque Panels", variable=self.selected_radio,
                                               value="Semi-Opaque Panels")

        radio_opaque.pack(anchor='w')
        radio_semiopaque.pack(anchor='w')

        rad_submit_button = tkinter.Button(radio_win, text='Submit',
                                       command=lambda: self.click_submit_radio(radio_win))
        rad_submit_button.pack(side='bottom',pady=5)

        radio_win.geometry("400x250")
        radio_win.mainloop()

        if self.best_model is None or self.scaler is None:
            print("No panel type selected, aborting...")
            return

        for file in files:
            try:
                preprocessor = Preprocessor(file)
                sheet_names = pd.read_excel(file, sheet_name=None)

                sheet_names_to_process = []

                self.make_listbox(sheet_names)

                for sheet in self.sheet_names_to_process:
                    scaled_data, df = preprocess_sheet(preprocessor, self.scaler, sheet)

                    # make predictions
                    predictions = self.best_model.predict(scaled_data)
                    predictions = [round(pred, 4) for pred in predictions]
                    df['EArray (KWh)'] = predictions

                    base_string = sheet.strip()
                    base_string = base_string.replace(" ", "").replace("_", "")
                    new_sheet = base_string + "Prediction"
                    if len(new_sheet) > 30:
                        diff = 30 - len("Prediction")
                        new_sheet = base_string[:diff] + "Prediction"
                    try:
                        with pd.ExcelWriter(file, engine='openpyxl', mode='a') as writer:
                            df.to_excel(writer, sheet_name=new_sheet, index=False)
                    except ValueError as e:
                        print(e)
                        print("Trying next sheet")

            except FileNotFoundError:
                print("Error file " + file + " could not be found. Please check filepath and try again. \n")
            except PermissionError as permE:
                print(permE)
                print("Try closing file and running again")


opaqueGui = Gui()
opaqueGui.run()
