"""
File used to compute the prediction of a rating based on the song embeddings and the groove study data. It uses a Ridge regression model to predict the rating, which is fitted by cross-validation. The function estimate_rating returns the RMSE and R2 scores, the prediction and the ground truth ratings.
"""

import pathlib
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from scipy.special import kl_div
from sklearn.feature_selection import VarianceThreshold

from groovestim.utils.io_utils import DEFAULT_RATINGS

RATING_LABELS = {
    'groove': 'Groove',
    'Q1_dance': 'Dance',
    'Q2_listen': 'Listen',
    'Q4_party': 'Party',
    'Q3_beat': 'Beat',
    'Q5_rhythm': 'Rhythm',
    'Q6_disturbing': 'Disturbing',
}

def estimate_groove_rating(song_embeddings, groove_study_dataframe, rating_name, n_cross_val = 4, seed=0):
    """
    Estimate the rating based on the song embeddings and the groove study data. It uses a Ridge regression model to predict the rating, which is fitted by cross-validation.
    The ratings can be 'groove', 'Q1_dance', 'Q2_listen','Q4_party', 'Q3_beat', 'Q5_rhythm','Q6_disturbing'
    """
    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    def _format_annotated_dataset(song_embeddings, groove_study_dataframe, rating = 'groove'):
        """
        Format the dataset for supervised learning between song embeddings (X) and groove ratings or other variables from the table (y)
        """
        X = []
        y = []
        # Sort keys to ensure deterministic ordering across runs
        for song_id in sorted(song_embeddings.keys()):
            groove_row = groove_study_dataframe.loc[groove_study_dataframe['id']==song_id]
            
            X.append(song_embeddings[song_id][0])
            y.append(float(groove_row[rating].values[0]))#groove_row[col].iloc[0]) #Other code: groove_row[col].values[0]
        X = np.stack(X)
        y = np.stack(y)
        return X, y

    # Format the dataset
    embeddings_processed, ground_truth_ratings = _format_annotated_dataset(song_embeddings, groove_study_dataframe, rating = rating_name)

    # Ridge regressor to estimate results from embeddings
    estimator = Ridge(random_state=seed, alpha=0.2, solver='cholesky')

    embeddings_to_fit = embeddings_processed.copy()

    # Use a single explicit KFold splitter for deterministic splits
    cv_splitter = KFold(n_splits=n_cross_val, shuffle=True, random_state=seed)

    # Compute the averaged RMSE between predictions and ground truth ratings 
    RMSE_score = np.mean(cross_val_score(estimator, embeddings_to_fit, ground_truth_ratings, cv=cv_splitter, scoring='neg_root_mean_squared_error')) # explained_variance

    # Compute the averaged R2 between predictions and ground truth ratings
    R2_score = np.mean(cross_val_score(estimator, embeddings_to_fit, ground_truth_ratings, cv=cv_splitter, scoring='r2')) # explained_variance

    # Compute the predictions
    prediction = cross_val_predict(estimator, embeddings_to_fit, ground_truth_ratings, cv=cv_splitter)

    return RMSE_score, R2_score, prediction, ground_truth_ratings

def compute_ratings_scores(
    song_embeddings,
    groove_ratings_dataframe,
    ratings=DEFAULT_RATINGS,
    n_cross_val=4,
    seed=0,
):
    """
    Estimate every rating via cross-validated Ridge regression.

    Parameters
    ----------
    song_embeddings : dict
        Mapping song_id → embedding array (shape 1×D).
    groove_ratings_dataframe : pd.DataFrame
        The groove study annotations.
    ratings : list[str]
        Rating columns to predict.
    n_cross_val : int
        Number of cross-validation folds.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    results : dict[str, dict]
        Mapping rating_name → {"RMSE": float, "R2": float,
        "prediction": np.ndarray, "ground_truth": np.ndarray}.
    """
    print("Estimating groove ratings...")

    results = {}
    for rating_name in ratings:
        RMSE_score, R2_score, prediction, ground_truth_ratings = (
            estimate_groove_rating(
                song_embeddings=song_embeddings,
                groove_study_dataframe=groove_ratings_dataframe,
                rating_name=rating_name,
                n_cross_val=n_cross_val,
                seed=seed,
            )
        )
        results[rating_name] = {
            "RMSE": RMSE_score,
            "R2": R2_score,
            "prediction": prediction,
            "ground_truth": ground_truth_ratings,
        }

    return results

def variance_threshold_embeddings(embeddings, var_thr):
    """
    Variance Threshold Selection, i.e. selecting features based on their variance
    """

    print(f"Original features shape: {embeddings.shape}")
    selector = VarianceThreshold(threshold=var_thr)

    try:
        embeddings_to_fit = selector.fit_transform(embeddings_to_fit)
    except ValueError:
        embeddings_to_fit = embeddings_to_fit
        print("No feature selection applied, none met the criterion")
        
    print(f"Features shape after thresholding: {embeddings_to_fit.shape}")
    return embeddings_to_fit

def save_scatter(ground_truth, predictions, rating_name="Rating", save_path=None,
                 styles=None, R2_score=None,
                 show_title=True, show_xlabel=True, show_ylabel=True):
    """
    Visualize the predicted ratings against the ground truth ratings in a 2D scatter plot.

    A linear regression line (OLS) with a 95 % confidence band is drawn on top of the
    scatter points. An optional per-style marker grouping is supported via ``styles``.
    Saves the figure to ``save_path`` when provided.
    """
    from groovestim.ratings.compute_viz import _MPL_MARKERS
    from scipy import stats

    ground_truth = np.asarray(ground_truth, dtype=float)
    predictions  = np.asarray(predictions,  dtype=float)

    # ── OLS regression ────────────────────────────────────────────────────
    slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth, predictions)
    x_fit   = np.linspace(ground_truth.min(), ground_truth.max(), 300)
    y_fit   = slope * x_fit + intercept

    # # 95 % confidence interval on the mean response
    # n       = len(ground_truth)
    # t_crit  = stats.t.ppf(0.975, df=n - 2)
    # x_mean  = ground_truth.mean()
    # se_band = std_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / np.sum((ground_truth - x_mean) ** 2))
    # y_lo    = y_fit - t_crit * se_band
    # y_hi    = y_fit + t_crit * se_band

    # ── Figure ────────────────────────────────────────────────────────────
    PALETTE = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(7, 6))

    if styles is not None:
        unique_styles = sorted(set(styles))
        style_to_marker = {
            s: _MPL_MARKERS[i % len(_MPL_MARKERS)] for i, s in enumerate(unique_styles)
        }
        styles_arr = np.array(styles)
        for i, style_name in enumerate(unique_styles):
            mask = styles_arr == style_name
            ax.scatter(
                ground_truth[mask], predictions[mask],
                marker=style_to_marker[style_name],
                color=PALETTE[i % len(PALETTE)],
                edgecolors="#555555", linewidths=0.4,
                s=60, alpha=0.85, label=style_name, zorder=3,
            )
    else:
        ax.scatter(
            ground_truth, predictions,
            marker="o", color=PALETTE[0],
            edgecolors="#555555", linewidths=0.4,
            s=60, alpha=0.85, label="Predictions", zorder=3,
        )

    # # Confidence band
    # ax.fill_between(x_fit, y_lo, y_hi, color="#9b59b6", alpha=0.18, zorder=1)

    # Regression line
    reg_label = f"Linear regression" #OLS fit  (slope={slope:.2f}, intercept={intercept:.2f})"
    ax.plot(x_fit, y_fit, color="#8e44ad", linewidth=2, label=reg_label, zorder=2)

    if show_xlabel:
        ax.set_xlabel(f"Ground Truth {RATING_LABELS[rating_name]}", fontsize=11)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(f"Predicted {RATING_LABELS[rating_name]}", fontsize=11)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Force both axes to share the same range and aspect ratio
    all_vals = np.concatenate([ground_truth, predictions])
    ax_min, ax_max = all_vals.min(), all_vals.max()
    padding = (ax_max - ax_min) * 0.05
    ax.set_xlim(ax_min - padding, ax_max + padding)
    ax.set_ylim(ax_min - padding, ax_max + padding)
    ax.set_aspect("equal", adjustable="box")

    if show_title:
        ax.set_title(f"Rating: {RATING_LABELS[rating_name]} | Predicted vs Ground Truth",
                     fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        print("Figure not saved, path is None")

    plt.close(fig)


def save_scatter_interactive(ground_truth, predictions, groove_ratings_dataframe,
                             rating_name="Rating", save_path=None, R2_score=None):
    """
    Interactive Plotly scatter plot of predicted vs ground-truth ratings.

    Includes an OLS regression line with a 95 % confidence band and a dashed
    ideal-diagonal reference.  Hover tooltips display the song name and the
    rating value, grouped by musical style (marker symbol), following the same
    convention as ``compute_viz.save_interactive``.
    """
    from groovestim.ratings.compute_viz import _PLOTLY_SYMBOLS
    from scipy import stats

    ground_truth = np.asarray(ground_truth, dtype=float)
    predictions  = np.asarray(predictions,  dtype=float)

    # ── OLS regression ────────────────────────────────────────────────────
    slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth, predictions)
    x_fit  = np.linspace(ground_truth.min(), ground_truth.max(), 300)
    y_fit  = slope * x_fit + intercept

    # 95 % confidence band
    n      = len(ground_truth)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    x_mean = ground_truth.mean()
    se_band = std_err * np.sqrt(
        1 / n + (x_fit - x_mean) ** 2 / np.sum((ground_truth - x_mean) ** 2)
    )
    y_lo = y_fit - t_crit * se_band
    y_hi = y_fit + t_crit * se_band

    # ── Song / style metadata ─────────────────────────────────────────────
    rating_col = rating_name.lower()  # e.g. "Groove" → "groove"
    df = groove_ratings_dataframe.drop_duplicates(subset="id").copy()

    song_names = []
    styles = []
    for gt_val in ground_truth:
        match = df.loc[np.isclose(df[rating_col].astype(float), gt_val)]
        if len(match) > 0:
            row = match.iloc[0]
            song_names.append(str(row["song"]) if "song" in df.columns else str(row["id"]))
            styles.append(str(row["style_family"]) if "style_family" in df.columns else "unknown")
            df = df.drop(row.name)
        else:
            song_names.append("unknown")
            styles.append("unknown")

    # Style → Plotly marker symbol mapping
    unique_styles = sorted(set(styles))
    style_to_symbol = {
        s: _PLOTLY_SYMBOLS[i % len(_PLOTLY_SYMBOLS)]
        for i, s in enumerate(unique_styles)
    }

    # Per-point hover text
    hover_texts = [
        "<br>".join([
            f"<b>{song_names[i]}</b>",
            f"style: {styles[i]}",
            f"ground truth {RATING_LABELS[rating_name]}: {ground_truth[i]:.2f}",
            f"predicted {RATING_LABELS[rating_name]}: {predictions[i]:.2f}",
        ])
        for i in range(n)
    ]

    # ── Build figure ──────────────────────────────────────────────────────
    fig = go.Figure()

    # 95 % CI band (filled area, no legend entry)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_fit, x_fit[::-1]]),
            y=np.concatenate([y_hi,  y_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(232, 121, 249, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        )
    )

    # Regression line
    reg_label = f"OLS fit (slope={slope:.2f}, intercept={intercept:.2f})"
    if p_value < 0.001:
        reg_label += "  p<0.001"
    else:
        reg_label += f"  p={p_value:.3f}"
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            line=dict(color="#e879f9", width=2.5),
            name=reg_label,
        )
    )

    # Ideal diagonal (y = x)
    all_vals = np.concatenate([ground_truth, predictions])
    diag_lo, diag_hi = all_vals.min(), all_vals.max()
    fig.add_trace(
        go.Scatter(
            x=[diag_lo, diag_hi],
            y=[diag_lo, diag_hi],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1.2),
            name="Ideal (y = x)",
        )
    )

    # One scatter trace per style
    for style_name in unique_styles:
        mask = np.array([s == style_name for s in styles])
        fig.add_trace(
            go.Scatter(
                x=ground_truth[mask],
                y=predictions[mask],
                mode="markers",
                marker=dict(
                    size=9,
                    opacity=0.85,
                    symbol=style_to_symbol[style_name],
                    line=dict(width=0.5, color="white"),
                ),
                text=[t for t, m in zip(hover_texts, mask) if m],
                hoverinfo="text",
                name=style_name,
            )
        )

    # ── Layout ────────────────────────────────────────────────────────────
    title = f"Predicted vs Ground Truth | Rating: {RATING_LABELS[rating_name]}"
    if R2_score is not None:
        title += f"<br><sup>R² = {R2_score:.4f}</sup>"

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16, color="white"), x=0.5, xanchor="center"),
        xaxis=dict(
            title=f"Ground Truth {RATING_LABELS[rating_name]}",
            gridcolor="#333355",
            showline=True, linecolor="#444466",
        ),
        yaxis=dict(
            title=f"Predicted {RATING_LABELS[rating_name]}",
            gridcolor="#333355",
            showline=True, linecolor="#444466",
        ),
        legend=dict(
            bgcolor="rgba(20,20,40,0.7)",
            bordercolor="#444466",
            borderwidth=1,
            font=dict(size=11),
        ),
        height=560,
        width=660,
        margin=dict(t=80, b=50, l=60, r=20),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
    )

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive figure saved to {save_path}")

    return fig
