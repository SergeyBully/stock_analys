import customtkinter as ctk
from CTkDatePicker import CTkDatePicker
from dateutil.relativedelta import relativedelta
from datetime import datetime
from plots import generate_plot

class GUI:
    def input_frame(self, root):
        # Frame for ticker, interval and start/end date
        self.input_frame = ctk.CTkFrame(root)
        self.input_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsw')

        # Ticker label
        self.label_ticker = ctk.CTkLabel(self.input_frame, text="Ticker:")
        self.label_ticker.grid(row=0, column=0, padx=5, pady=5)
        # Ticker Dropdown (Combobox)
        self.ticker_options = ["NVDA", "XLF", "INTC", "MSFT", "QS", "TSM"]
        self.ticker_var = ctk.StringVar(value="NVDA")
        self.dropdown_ticker = ctk.CTkComboBox(self.input_frame, values=self.ticker_options,variable=self.ticker_var)
        self.dropdown_ticker.grid(row=0, column=1, padx=5, pady=5)

        # Interval Label
        self.label_interval = ctk.CTkLabel(self.input_frame, text="Interval:")
        self.label_interval.grid(row=1, column=0, padx=5, pady=5)
        # Interval Dropdown (Combobox)
        self.interval_options = ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        self.interval_var = ctk.StringVar(value="1d")
        self.dropdown_interval = ctk.CTkComboBox(self.input_frame, values=self.interval_options, variable=self.interval_var)
        self.dropdown_interval.grid(row=1, column=1, padx=5, pady=5)

        # Start Date label and Date Picker
        self.label_start = ctk.CTkLabel(self.input_frame, text="Start Date (YYYY-MM-DD):")
        self.label_start.grid(row=2, column=0, padx=5, pady=5)
        self.entry_start = CTkDatePicker(self.input_frame)
        self.entry_start.grid(row=2, column=1, padx=5, pady=5)
        self.entry_start.set_insert_date(0,
                                    (datetime.today() - relativedelta(months=3)).strftime("%Y-%m-%d"))  # 3 months ago

        # End Date label and Date Picker
        self.label_end = ctk.CTkLabel(self.input_frame, text="End Date (YYYY-MM-DD):")
        self.label_end.grid(row=3, column=0, padx=5, pady=5)
        self.entry_end = CTkDatePicker(self.input_frame)
        self.entry_end.grid(row=3, column=1, padx=5, pady=5)
        self.entry_end.set_insert_date(0, datetime.today().strftime("%Y-%m-%d"))  # Today date


    def checkbox_indicators_frame(self, root):

        # Frame for indicator checkboxes
        self.checkbox_indicators_frame = ctk.CTkFrame(root)
        self.checkbox_indicators_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsw')

        # Bullish/Bearish checkbox
        self.show_bullish_bearish = ctk.BooleanVar(value=False)
        self.checkbox_bullish_bearish = ctk.CTkCheckBox(self.checkbox_indicators_frame, text="Show Bullish/Bearish",
                                                        variable=self.show_bullish_bearish)
        self.checkbox_bullish_bearish.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        # Fibonacci Retracement checkbox
        self.show_fibonacci_retracement = ctk.BooleanVar(value=False)
        self.checkbox_fibonacci_retracement = ctk.CTkCheckBox(self.checkbox_indicators_frame, text="Show Fibonacci Retracement",
                                                        variable=self.show_fibonacci_retracement)
        self.checkbox_fibonacci_retracement.grid(row=1, column=0, padx=5, pady=5, sticky='nw')

        # Bollinger Bands checkbox
        self.show_bollinger = ctk.BooleanVar(value=False)
        self.checkbox_bollinger = ctk.CTkCheckBox(self.checkbox_indicators_frame, text="Show Bollinger Bands", variable=self.show_bollinger)
        self.checkbox_bollinger.grid(row=2, column=0, padx=5, pady=5, sticky='nw')
        # Bollinger Bands window entry
        self.entry_window_bollinger = ctk.CTkEntry(self.checkbox_indicators_frame, width=self.subentry_size, placeholder_text="Window")
        self.entry_window_bollinger.grid(row=2, column=1, padx=5, pady=5)
        self.entry_window_bollinger.insert(0, "20")

        # Moving Averages checkbox
        self.show_moving_averages = ctk.BooleanVar(value=True)
        self.checkbox_moving_averages = ctk.CTkCheckBox(self.checkbox_indicators_frame, text="Show Moving Averages\n(SMA, EMA, WMA)",
                                                        variable=self.show_moving_averages)
        self.checkbox_moving_averages.grid(row=3, column=0, padx=5, pady=5, sticky='nw')
        # Moving Averages window entry
        self.entry_window_moving_averages = ctk.CTkEntry(self.checkbox_indicators_frame, width=self.subentry_size, placeholder_text="Window")
        self.entry_window_moving_averages.grid(row=3, column=1, padx=5, pady=5)
        self.entry_window_moving_averages.insert(0, "5")

        # Relative Strength Index checkbox
        self.show_relative_strength = ctk.BooleanVar(value=True)
        self.checkbox_relative_strength = ctk.CTkCheckBox(self.checkbox_indicators_frame, text="Show Relative Strength Index",
                                                        variable=self.show_relative_strength)
        self.checkbox_relative_strength.grid(row=4, column=0, padx=5, pady=5, sticky='nw')
        # Relative Strength Index window entry
        self.entry_window_relative_strength = ctk.CTkEntry(self.checkbox_indicators_frame, width=self.subentry_size, placeholder_text="Window")
        self.entry_window_relative_strength.grid(row=4, column=1, padx=5, pady=5)
        self.entry_window_relative_strength.insert(0, "14")


    def checkbox_ML_frame(self, root):


        # Frame for ML checkboxes
        self.checkbox_ML_frame = ctk.CTkFrame(root)
        self.checkbox_ML_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsw')

        # Random Forest Regressor checkbox
        self.show_forest_regression = ctk.BooleanVar(value=False)
        self.checkbox_forest_regression = ctk.CTkCheckBox(self.checkbox_ML_frame, text="Show Random Forest Regressor",
                                                        variable=self.show_forest_regression)
        self.checkbox_forest_regression.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        # Period/steps window entry
        self.entry_periods = ctk.CTkEntry(self.checkbox_ML_frame, width=self.subentry_size, placeholder_text="Window")
        self.entry_periods.grid(row=0, column=1, padx=5, pady=5)
        self.entry_periods.insert(0, "5")

    def button_frame(self, root):
        self.button_frame = ctk.CTkFrame(root)
        self.button_frame.grid(row=1, column=0, padx=20, pady=20, sticky='es')
        self.button = ctk.CTkButton(self.button_frame, text="Show Graph",
                               command=lambda: generate_plot(self.dropdown_ticker.get(), self.entry_start.get_date(), self.entry_end.get_date(), self.dropdown_interval.get(), \
                                                             {"bullish_bearish": self.show_bullish_bearish.get(),
                                                                       "fibonacci_retracement": self.show_fibonacci_retracement.get(),
                                                                       "bollinger_bands": {
                                                                            "show":self.show_bollinger.get(),
                                                                            "window":self.entry_window_bollinger.get()},
                                                                       "moving_averages": {
                                                                            "show":self.show_moving_averages.get(),
                                                                             "window":self.entry_window_moving_averages.get()},
                                                                       "relative_strength": {
                                                                             "show": self.show_relative_strength.get(),
                                                                             "window": self.entry_window_relative_strength.get()},
                                                                       "forest_regression": {
                                                                             "show": self.show_forest_regression.get()}
                                                                       },
                                                             self.entry_periods.get()))
        self.button.grid(row=0, column=0, columnspan=2)

    def __init__(self, root):
        self.subentry_size = 60
        # Init all frames
        self.input_frame(root)
        self.checkbox_indicators_frame(root)
        self.checkbox_ML_frame(root)
        self.button_frame(root)



def start_GUI():
    root = ctk.CTk()
    root.title("Stock Price Viewer")
    create_GUI = GUI(root)

    # Window start
    root.mainloop()