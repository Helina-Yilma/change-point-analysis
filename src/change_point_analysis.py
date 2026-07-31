import logging
from typing import Optional, Tuple, Dict, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class BrentOilChangePointAnalysis:
    """
    Bayesian change point detection for Brent oil prices.
    
    This class provides a complete pipeline for loading, preparing, and analyzing
    Brent crude oil prices, identifying potential regime shifts in daily log returns
    using a Bayesian change point model.

    Attributes:
        csv_path (str): Path to CSV containing Brent oil prices.
        events_csv (Optional[str]): Path to CSV containing relevant events.
        data (Optional[pd.DataFrame]): Loaded and processed price data.
        log_returns (Optional[np.ndarray]): Daily log returns array.
        model (Optional[pm.Model]): PyMC model object.
        trace (Optional[az.InferenceData]): Posterior trace of the model.
        tau_posterior (Optional[np.ndarray]): Posterior samples for the change point index.
        events (Optional[pd.DataFrame]): Event data, if provided.
    """

    def __init__(self, csv_path: str, events_csv: Optional[str] = None) -> None:
        """
        Initialize BrentOilChangePointAnalysis instance.

        Args:
            csv_path (str): Path to Brent oil price CSV with 'Date' and 'Price' columns.
            events_csv (Optional[str]): Path to CSV with events (optional).
        """
        self.csv_path = csv_path
        self.events_csv = events_csv
        self.data: Optional[pd.DataFrame] = None
        self.log_returns: Optional[np.ndarray] = None
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.tau_posterior: Optional[np.ndarray] = None
        self.events: Optional[pd.DataFrame] = None

    # -------------------------
    # Data Preparation & EDA
    # -------------------------
    def load_and_prepare_data(self) -> None:
        """
        Load CSV price data, parse dates, compute daily log returns, 
        and thin the dataset for performance.
        
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing.
        """
        logging.info("Loading and preparing data.")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path)
        if {"Date", "Price"} - set(self.data.columns):
            raise ValueError("CSV must contain 'Date' and 'Price' columns.")

        self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce", format="mixed")
        self.data = self.data.dropna().sort_values("Date").reset_index(drop=True)

        # Performance optimization: thin dataset to reduce computation
        self.data = self.data.iloc[::5].reset_index(drop=True)

        # Compute daily log returns
        self.data["LogPrice"] = np.log(self.data["Price"])
        self.data["LogReturn"] = self.data["LogPrice"].diff()
        self.log_returns = self.data["LogReturn"].dropna().values

        logging.info(
            f"Prepared {len(self.log_returns)} daily log-return observations after thinning."
        )

    def plot_price_series(self, overlay_tau: bool = False) -> None:
        """
        Plot the historical Brent oil price series.

        Args:
            overlay_tau (bool): If True, overlays the median change point as a red line.
        """
        if self.data is None:
            logging.error("No data available for plotting. Run load_and_prepare_data first.")
            return

        plt.figure(figsize=(12, 5))
        plt.plot(self.data["Date"], self.data["Price"], label="Price")
        if overlay_tau and self.tau_posterior is not None:
            tau_idx = int(np.median(self.tau_posterior))
            plt.axvline(self.data["Date"].iloc[tau_idx + 1], color='red', linestyle='--', label='Change Point')
        plt.title("Brent Oil Price (USD per Barrel)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def plot_log_returns(self) -> None:
        """
        Plot daily log returns of Brent oil prices.
        """
        if self.log_returns is None:
            logging.error("No log return data available. Run load_and_prepare_data first.")
            return

        plt.figure(figsize=(12, 5))
        plt.plot(self.data["Date"].iloc[1:], self.log_returns)
        plt.title("Daily Log Returns of Brent Oil Prices")
        plt.xlabel("Date")
        plt.ylabel("Log Return")
        plt.show()

    # -------------------------
    # Bayesian Change Point Model
    # -------------------------
    def build_and_run_model(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        seed: int = 42
    ) -> None:
        """
        Build a Bayesian change point model and sample posterior distributions.

        Args:
            draws (int): Number of posterior draws.
            tune (int): Number of tuning steps.
            chains (int): Number of MCMC chains.
            target_accept (float): Step size adaptation target.
            seed (int): Random seed for reproducibility.

        Raises:
            RuntimeError: If log returns data is not prepared.
        """
        if self.log_returns is None:
            raise RuntimeError("Prepare data before modeling using load_and_prepare_data.")

        logging.info("Building Bayesian change point model.")

        n = len(self.log_returns)
        time_index = np.arange(n)

        with pm.Model() as model:

            # Narrow prior for faster convergence
            tau = pm.DiscreteUniform("tau", lower=int(0.2*n), upper=int(0.8*n))
            mu_1 = pm.Normal("mu_1", mu=0.0, sigma=0.01)
            mu_2 = pm.Normal("mu_2", mu=0.0, sigma=0.01)
            sigma = pm.HalfNormal("sigma", sigma=0.05)

            mu = pm.math.switch(time_index < tau, mu_1, mu_2)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self.log_returns)

            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                return_inferencedata=True,
                progressbar=True
            )

        self.model = model
        self.tau_posterior = self.trace.posterior["tau"].values.flatten()
        logging.info("Bayesian model sampling completed.")

    # -------------------------
    # Diagnostics
    # -------------------------
    def check_convergence(self) -> None:
        """
        Print summary and trace plots of posterior samples to check convergence.
        """
        if self.trace is None:
            logging.error("No posterior trace available. Run build_and_run_model first.")
            return

        summary = az.summary(self.trace)
        print(summary)
        az.plot_trace(self.trace)
        plt.show()

    # -------------------------
    # Interpretation
    # -------------------------
    def plot_tau_posterior(self) -> None:
        """
        Plot histogram of change point posterior distribution.
        """
        if self.tau_posterior is None:
            logging.error("Tau posterior not available. Run build_and_run_model first.")
            return

        plt.figure(figsize=(10, 4))
        plt.hist(self.tau_posterior, bins=40, density=True)
        plt.title("Posterior Distribution of Change Point (tau)")
        plt.xlabel("Time Index")
        plt.ylabel("Density")
        plt.show()

    def plot_mu_posteriors(self) -> None:
        """
        Plot histograms of mean log return posterior distributions.
        """
        if self.trace is None:
            logging.error("Posterior trace not available. Run build_and_run_model first.")
            return

        mu1_samples = self.trace.posterior["mu_1"].values.flatten()
        mu2_samples = self.trace.posterior["mu_2"].values.flatten()
        plt.figure(figsize=(10, 5))
        plt.hist(mu1_samples, bins=50, alpha=0.6, label="mu_1 (Before)")
        plt.hist(mu2_samples, bins=50, alpha=0.6, label="mu_2 (After)")
        plt.title("Posterior Distributions of Mean Log Returns")
        plt.xlabel("Log Return")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def get_change_point_date(self) -> Tuple[int, pd.Timestamp]:
        """
        Retrieve the median change point index and corresponding date.

        Returns:
            Tuple[int, pd.Timestamp]: Change point index and date.

        Raises:
            RuntimeError: If tau posterior is not available.
        """
        if self.tau_posterior is None or self.data is None:
            raise RuntimeError("Tau posterior or data unavailable.")
        tau_median = int(np.median(self.tau_posterior))
        change_date = self.data["Date"].iloc[tau_median + 1]
        return tau_median, change_date

    def quantify_impact(self) -> Dict[str, float]:
        """
        Quantify the impact of the change point on mean log returns.

        Returns:
            Dict[str, float]: Mean log returns before and after, and percentage change.

        Raises:
            RuntimeError: If posterior trace is not available.
        """
        if self.trace is None:
            raise RuntimeError("Posterior trace unavailable.")
        mu1 = self.trace.posterior["mu_1"].values.flatten().mean()
        mu2 = self.trace.posterior["mu_2"].values.flatten().mean()
        pct_change = ((np.exp(mu2) - np.exp(mu1)) / np.exp(mu1)) * 100
        return {
            "mean_log_return_before": mu1,
            "mean_log_return_after": mu2,
            "percentage_change": pct_change
        }

    # -------------------------
    # Event Association
    # -------------------------
    def load_events(self) -> None:
        """
        Load events from CSV if provided, sort by date.

        Logs a warning if no CSV is provided.
        """
        if self.events_csv:
            if not os.path.exists(self.events_csv):
                logging.warning(f"Events CSV not found: {self.events_csv}")
                self.events = pd.DataFrame(columns=["event_date", "event_name", "description"])
                return
            self.events = pd.read_csv(self.events_csv)
            self.events["event_date"] = pd.to_datetime(self.events["event_date"])
            self.events = self.events.sort_values("event_date").reset_index(drop=True)
            logging.info("Loaded %d events from CSV.", len(self.events))
        else:
            logging.warning("No events CSV provided; event association skipped.")
            self.events = pd.DataFrame(columns=["event_date", "event_name", "description"])

    def associate_change_point_with_event(self) -> Optional[pd.Series]:
        """
        Find closest event prior to or on the detected change point.

        Returns:
            Optional[pd.Series]: Closest event row or None if no events match.
        """
        if self.events is None:
            self.load_events()

        _, change_date = self.get_change_point_date()
        prior_events = self.events[self.events["event_date"] <= change_date]
        if prior_events.empty:
            return None
        closest_event = prior_events.iloc[-1]
        return closest_event

    # -------------------------
    # Results Export
    # -------------------------
    def save_results_for_dashboard(
        self, 
        price_path: str = "brent_prices.csv", 
        change_point_path: str = "change_point_results.csv"
    ) -> None:
        """
        Save historical prices and change point results to CSV for dashboards.

        Args:
            price_path (str): Path to save price CSV.
            change_point_path (str): Path to save change point results CSV.

        Raises:
            RuntimeError: If data or tau posterior is unavailable.
        """
        if self.data is None or self.tau_posterior is None:
            raise RuntimeError("Task 2 results not available. Run full analysis first.")

        for path in [price_path, change_point_path]:
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Created directory {dir_path} for CSV.")

        self.data[['Date', 'Price']].to_csv(price_path, index=False)
        logging.info(f"Historical prices saved to {price_path}")

        tau_idx, change_date = self.get_change_point_date()
        impact = self.quantify_impact()
        cp_results = pd.DataFrame({
            'tau_index': [tau_idx],
            'change_date': [str(change_date.date())],
            'mean_log_return_before': [impact['mean_log_return_before']],
            'mean_log_return_after': [impact['mean_log_return_after']],
            'percentage_change': [impact['percentage_change']]
        })
        cp_results.to_csv(change_point_path, index=False)
        logging.info(f"Change point results saved to {change_point_path}")

    # -------------------------
    # Full Pipeline
    # -------------------------
    def run_full_analysis(self) -> None:
        """
        Run the complete analysis pipeline: load data, plot, model, diagnose,
        interpret, and associate with events.
        """
        self.load_and_prepare_data()
        self.plot_price_series()
        self.plot_log_returns()
        self.build_and_run_model()
        self.check_convergence()
        self.plot_tau_posterior()
        self.plot_mu_posteriors()
        self.plot_price_series(overlay_tau=True)

        tau_idx, date = self.get_change_point_date()
        impact = self.quantify_impact()

        print(
            f"\nChange point detected around {date.date()}.\n"
            f"Mean daily log return shifted from "
            f"{impact['mean_log_return_before']:.5f} to "
            f"{impact['mean_log_return_after']:.5f}, "
            f"representing a {impact['percentage_change']:.2f}% change."
        )

        event = self.associate_change_point_with_event()
        if event is not None:
            print(
                f"Closest relevant event prior to change point:\n"
                f"{event['event_name']} on {event['event_date'].date()}: {event['description']}"
            )
        else:
            print("No relevant event found near the change point.")
