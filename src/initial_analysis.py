from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class BrentOilAnalysisFoundation:
    """
    Foundation class for Task 1: laying the groundwork for Brent oil price analysis.

    Responsibilities:
    - Load and validate Brent oil price data.
    - Compute basic time series diagnostics (trend, returns, volatility).
    - Check stationarity of the price series.
    - Compile major geopolitical, OPEC, and macroeconomic events.
    - Save structured outputs needed for downstream analysis.
    """

    def __init__(self, brent_data_path: Path, events_output_path: Path) -> None:
        self.brent_data_path: Path = brent_data_path
        self.events_output_path: Path = events_output_path
        self.brent_df: Optional[pd.DataFrame] = None

    def load_brent_data(self) -> pd.DataFrame:
        """
        Load and validate the Brent oil price dataset.

        Returns
        -------
        pd.DataFrame
            Cleaned Brent oil price data with Date as index.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If required columns are missing.
        """
        logging.info("Loading Brent oil price data...")
        if not self.brent_data_path.exists():
            raise FileNotFoundError(f"Brent data file not found: {self.brent_data_path}")

        df = pd.read_csv(self.brent_data_path)
        required_cols = {"Date", "Price"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Brent data must contain columns: {required_cols}, found: {set(df.columns)}")

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed')
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").set_index("Date")

        self.brent_df = df
        logging.info("Brent data loaded successfully with %d rows.", len(df))
        return df

    def analyze_time_series_properties(self) -> pd.DataFrame:
        """
        Compute time series diagnostics and visualize trends.

        Diagnostics include:
        - Log price
        - Price differences
        - Log returns
        - Rolling volatility

        Returns
        -------
        pd.DataFrame
            DataFrame including diagnostic columns.
        """
        if self.brent_df is None:
            raise RuntimeError("Brent data must be loaded before analysis.")

        logging.info("Analyzing time series properties...")
        df = self.brent_df.copy()
        df["log_price"] = df["Price"].apply(lambda x: np.nan if x <= 0 else np.log(x))
        df["price_diff"] = df["Price"].diff()
        df["log_return"] = df["log_price"].diff()
        df["rolling_volatility"] = df["log_return"].rolling(30).std()

        # Plot price trend
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df["Price"], label="Price")
        plt.title("Brent Oil Price")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot rolling volatility
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df["rolling_volatility"], label="30-day Rolling Volatility", color="orange")
        plt.title("Brent Oil Price Rolling Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.grid(True)
        plt.legend()
        plt.show()

        return df


    def check_stationarity(self, df: pd.DataFrame) -> None:
        """
        Check stationarity of the log price series with one plot.

        - Performs Augmented Dickey-Fuller (ADF) test
        - Plots log price with rolling mean & rolling std

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'log_price' column.
        """
        if "log_price" not in df.columns:
            raise ValueError("DataFrame must contain 'log_price' column.")

        logging.info("Performing stationarity check (ADF test)...")
        log_price = df["log_price"].dropna()

        # ADF test
        adf_result = adfuller(log_price)
        logging.info("ADF Statistic: %.4f", adf_result[0])
        logging.info("p-value: %.4f", adf_result[1])
        if adf_result[1] < 0.05:
            logging.info("Series appears stationary (reject H0)")
        else:
            logging.info("Series appears non-stationary (fail to reject H0)")

        # Single stationarity plot
        rolmean = log_price.rolling(30).mean()
        rolstd = log_price.rolling(30).std()

        plt.figure(figsize=(12, 5))
        plt.plot(log_price.index, log_price, color="blue", label="Log Price")
        plt.plot(rolmean.index, rolmean, color="red", label="30-day Rolling Mean")
        plt.plot(rolstd.index, rolstd, color="green", label="30-day Rolling Std")
        plt.title("Stationarity Check: Rolling Mean & Std")
        plt.xlabel("Date")
        plt.ylabel("Log Price")
        plt.legend()
        plt.grid(True)
        plt.show()


    def define_relevant_events(self) -> pd.DataFrame:
        """
        Define and compile a structured dataset of major events affecting oil prices.

        Returns
        -------
        pd.DataFrame
            Events dataset with columns: event_date, event_name, description
        """
        logging.info("Defining major geopolitical and economic events...")
        events = [
            {"event_date": "1973-10-17", "event_name": "Arab Oil Embargo", "description": "OAPEC embargo causes supply shock."},
            {"event_date": "1979-01-01", "event_name": "Iranian Revolution", "description": "Disruption of Iranian oil production."},
            {"event_date": "1990-08-02", "event_name": "Iraqi Invasion of Kuwait", "description": "Severe supply disruption."},
            {"event_date": "1997-07-01", "event_name": "Asian Financial Crisis", "description": "Demand shock from collapsing Asian economies."},
            {"event_date": "2001-09-11", "event_name": "September 11 Attacks", "description": "Heightened geopolitical uncertainty."},
            {"event_date": "2008-09-15", "event_name": "Global Financial Crisis", "description": "Sharp collapse in demand and prices."},
            {"event_date": "2010-04-20", "event_name": "Deepwater Horizon Spill", "description": "Offshore production concerns."},
            {"event_date": "2014-11-27", "event_name": "OPEC Maintains Production", "description": "Market share over price support."},
            {"event_date": "2016-11-30", "event_name": "OPEC Production Cut Agreement", "description": "Coordinated output cuts."},
            {"event_date": "2020-03-11", "event_name": "COVID-19 Pandemic Declared", "description": "Historic collapse in demand."},
            {"event_date": "2020-04-20", "event_name": "Negative Oil Prices", "description": "WTI prices below zero."},
            {"event_date": "2022-02-24", "event_name": "Russian Invasion of Ukraine", "description": "Sanctions and supply risks."},
            {"event_date": "2023-04-02", "event_name": "OPEC+ Surprise Production Cuts", "description": "Unexpected cuts trigger price reaction."},
            {"event_date": "2004-11-16", "event_name": "Iraq Oil-for-Food Program Investigation", "description": "U.S. Senate investigation reveals irregularities in Iraqi oil sales, increasing supply uncertainty"}

        ]
        events_df = pd.DataFrame(events)
        events_df["event_date"] = pd.to_datetime(events_df["event_date"])
        events_df = events_df.sort_values("event_date")
        logging.info("Compiled %d major events.", len(events_df))
        return events_df

    def save_events_dataset(self, events_df: pd.DataFrame) -> None:
        """
        Save the compiled events dataset to disk.

        Parameters
        ----------
        events_df : pd.DataFrame
            Structured events dataset.
        """
        logging.info("Saving events dataset to %s", self.events_output_path)
        self.events_output_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(self.events_output_path, index=False)
        logging.info("Events dataset saved successfully.")

    def run_task_1_pipeline(self) -> None:
        """
        Execute the full Task 1 preparation pipeline:
        - Load Brent data
        - Analyze time series properties
        - Compile and save events dataset
        """
        self.load_brent_data()
        self.analyze_time_series_properties()
        events_df = self.define_relevant_events()
        self.save_events_dataset(events_df)
        logging.info("Task 1 pipeline completed successfully.")


# -------------------------------
# Example execution
# -------------------------------
if __name__ == "__main__":
    foundation = BrentOilAnalysisFoundation(
        brent_data_path=Path("../data/raw/BrentOilPrices.csv"),
        events_output_path=Path("../data/processed/geopolitical_events.csv")
    )
    foundation.run_task_1_pipeline()
