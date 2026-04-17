from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

#ui def
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2("Trader Controls"),
        ui.input_select(
            "stock_id", 
            "Select Target Asset (Stock ID):", 
            choices=[str(i) for i in range(127)] 
        ),
        ui.hr(),
        ui.p("Methodology Pipeline:"),
        ui.tags.ul(
            ui.tags.li("M1 (Gia): DuckDB Data Engineering"),
            ui.tags.li("M2 (Jamie): EDA & WLS Regression"),
            ui.tags.li("M3 (Rosa): HAV-RV / HAR-RV"),
            ui.tags.li("M4 (Jisu): ARMA-GARCH")
        ),
        ui.hr(),
        ui.p("Select an asset to dynamically filter all diagnostic evaluations.")
    ),
    
    ui.h1("High-Frequency Volatility Prediction Pipeline"),
    
    ui.navset_card_tab(
        
        #m5, evaluation 
        ui.nav_panel("Executive Evaluation", 
            ui.h3("Model Competition: Predictive Performance"),
            ui.p("Evaluating out-of-sample QLIKE and MSE. QLIKE heavily penalizes under-prediction of volatility."),
            ui.output_table("master_metrics_table"),
            ui.hr(),
            ui.h3("Predicted vs. Actual Realized Volatility"),
            ui.output_plot("master_prediction_plot")
        ),

        #EDA and market regimes tab (m2)
        ui.nav_panel("M2: Market Regimes (EDA)",
            ui.h3("Exploratory Data Analysis & Clustering"),
            ui.div(
                {"style": "border: 2px dashed #17a2b8; border-radius: 10px; padding: 40px; text-align: center; background-color: #f8f9fa;"},
                ui.h4("Awaiting M2 (Jamie) Data Contract Fulfillment"),
                ui.p("Required files: m2_eda_summary.csv / cluster_plots.png"),
                ui.p("Reserved for volatility clustering visualizations and bid-ask spread interactions.")
            )
        ),
        
        #HAR-RV diagnostics tab (m3)
        ui.nav_panel("M3: HAV-RV Diagnostics",
            ui.h3("Heterogeneous Autoregressive Model Fitness"),
            ui.div(
                {"style": "border: 2px dashed #28a745; border-radius: 10px; padding: 20px; text-align: center; background-color: #f8f9fa; margin-bottom: 20px;"},
                ui.h4("Awaiting M3 (Rosa) Data Contract Fulfillment"),
                ui.p("Required file: m3_model_comparison.csv"),
                ui.p("Reserved for the table comparing HAV-RV FULL vs. SHORT vs. HAV-RV-X.")
            ),
            ui.h4("Beta Estimates & 95% Confidence Intervals"),
            ui.div(
                {"style": "border: 2px dashed #28a745; border-radius: 10px; padding: 40px; text-align: center; background-color: #f8f9fa;"},
                ui.h4("Awaiting M3 (Rosa) Data Contract Fulfillment"),
                ui.p("Required file: m3_beta_estimates.csv"),
                ui.p("Will display the OLS/WLS parameter significance tables to prove physical lag mechanics.")
            )
        ),
        
        #ARMA-GARCH diagnostics tab (m4)
        ui.nav_panel("M4: ARMA-GARCH Diagnostics", 
            ui.h3("EGARCH-X Fit Information (AIC / BIC)"),
            ui.p("Proof of model fitness pulled directly from M4 outputs."),
            ui.output_table("m4_fit_table"),
            ui.hr(),
            
            # space for dynamic acf residual testing
            ui.h3("Residual Diagnostics: Autocorrelation Function (ACF)"),
            ui.output_plot("m4_acf_plot"),
            ui.hr(),

            # space for dynamic rv distribution
            ui.h3("Distribution of Realized Volatility"),
            ui.output_plot("m4_rv_dist_plot")
        )
    )
)

# --- 2. SERVER LOGIC ---
def server(input, output, session):
    
    # use robust absolute pathing for windows directory management
    m4_dir = Path(__file__).parent / "m4_outputs"
    
    # defensive data ingestion functions
    @reactive.Calc
    def get_m4_eval():
        eval_path = m4_dir / "garch_eval_results.csv"
        if eval_path.exists():
            return pd.read_csv(eval_path)
        return pd.DataFrame()

    @reactive.Calc
    def get_m4_fit():
        fit_path = m4_dir / "egarchx_fit_info.csv"
        if fit_path.exists():
            return pd.read_csv(fit_path)
        return pd.DataFrame({"System Notice": ["m4_outputs/egarchx_fit_info.csv missing from repository."]})

    # tab 1: master evaluation rendering
    @output
    @render.table
    def master_metrics_table():
        df_m4 = get_m4_eval()
        
        # mock m2 and m3 pending data structure
        df_pending = pd.DataFrame({
            "Model": ["WLS Regression (M2 - Pending)", "HAV-RV FULL WLS (M3 - Pending)"],
            "MSE": [np.nan, np.nan],
            "QLIKE": [np.nan, np.nan]
        })

        if not df_m4.empty and "stock_id" in df_m4.columns:
            df_m4["stock_id"] = df_m4["stock_id"].astype(str)
            filtered_m4 = df_m4[df_m4["stock_id"] == input.stock_id()].copy()
            
            if not filtered_m4.empty:
                filtered_m4["Model"] = "ARMA-GARCH (M4)"
                cols_to_keep = ["Model", "MSE", "QLIKE"]
                available_cols = [c for c in cols_to_keep if c in filtered_m4.columns]
                final_table = pd.concat([filtered_m4[available_cols], df_pending], ignore_index=True)
                return final_table
            else:
                # notify user which stocks were actually processed by m4
                available_stocks = ", ".join(df_m4["stock_id"].unique())
                return pd.DataFrame({"Notice": [f"Asset {input.stock_id()} not processed by M4 yet. Available test stocks: {available_stocks}"]})
                
        return pd.DataFrame({"Status": ["Awaiting aggregated prediction metrics CSVs from all tracks."]})

    @output
    @render.plot
    def master_prediction_plot():
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # text placeholder until prediction csvs are provided
        ax.text(0.5, 0.5, f"Awaiting 'predictions.csv' from M2, M3, M4\nReady to plot time-series for Asset {input.stock_id()}", 
                horizontalalignment='center', verticalalignment='center', fontsize=12, color="#6c757d")
        
        ax.set_title(f"Volatility Convergence Tracking: Asset {input.stock_id()}", fontweight="bold")
        ax.set_xlabel("Time Bucket (30s intervals)")
        ax.set_ylabel("Realized Volatility")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 0.01)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig

    # tab 4: m4 live rendering and dynamic placeholders
    @output
    @render.table
    def m4_fit_table():
        df = get_m4_fit()
        if "System Notice" in df.columns:
            return df
            
        if "stock_id" in df.columns:
            df["stock_id"] = df["stock_id"].astype(str)
            filtered = df[df["stock_id"] == input.stock_id()]
            if filtered.empty:
                available_stocks = ", ".join(df["stock_id"].unique())
                return pd.DataFrame({"Data Limitation": [f"Jisu has not computed ARMA-GARCH for Asset {input.stock_id()}. Available test stocks: {available_stocks}"]})
            return filtered
        return df.head(5)

    @output
    @render.plot
    def m4_acf_plot():
        # placeholder for the acf plot that will be dynamically generated once residuals are uploaded
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Awaiting 'm4_residuals.csv' to compute dynamic ACF for Asset {input.stock_id()}\nThis proves the model successfully captures volatility clustering (white noise check).", 
                horizontalalignment='center', verticalalignment='center', fontsize=11, color="#28a745")
        ax.set_axis_off()
        return fig

    @output
    @render.plot
    def m4_rv_dist_plot():
        # placeholder for the distribution comparison
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Awaiting 'm4_predictions.csv' to plot RV Distribution for Asset {input.stock_id()}\nVisual check of distribution tails and skewness.", 
                horizontalalignment='center', verticalalignment='center', fontsize=11, color="#28a745")
        ax.set_axis_off()
        return fig

app = App(app_ui, server)