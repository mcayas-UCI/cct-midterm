# Implementing Cultural Consensus Theory (CCT) with PyMC
# Code developed with AI assistance

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

def load_plant_knowledge_data(filepath):
    """
    Load binary response data from a CSV and return as a NumPy array.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing informant responses.
    
    Returns:
    --------
    numpy.ndarray
        Binary matrix of responses (shape: N informants x M questions).
    """
    df = pd.read_csv(filepath)
    data = df.iloc[:, 1:].values  # Skip the identifier column
    return data

def run_cct_model(data):
    """
    Fit a Cultural Consensus Theory model to binary response data using PyMC.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Binary response matrix (N informants x M questions).
    
    Returns:
    --------
    dict
        Dictionary containing model, trace, and summary statistics.
    """
    N, M = data.shape  # Number of informants and questions

    with pm.Model() as cct:
        # Prior for informant competence (scaled to range [0.5, 1])
        D_raw = pm.Beta("D_raw", alpha=2, beta=1, shape=N)
        D = pm.Deterministic("D", 0.5 + 0.5 * D_raw)

        # Prior for correct answers per question (unbiased prior)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Probability of informant agreeing with the true answer
        p = Z * D[:, None] + (1 - Z) * (1 - D[:, None])

        # Observed responses modeled as Bernoulli trials
        X = pm.Bernoulli("X", p=p, observed=data)

        # MCMC sampling
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            random_seed=42
        )

        summary = az.summary(trace)

    return {
        "model": cct,
        "trace": trace,
        "summary": summary
    }

def analyze_results(results, data):
    """
    Summarize and interpret posterior distributions from the model.
    
    Parameters:
    -----------
    results : dict
        Output dictionary from the model.
    data : numpy.ndarray
        Original binary response matrix.
    
    Returns:
    --------
    dict
        Aggregated analysis metrics and comparison outputs.
    """
    trace = results["trace"]
    posterior = trace.posterior

    # Posterior mean of informant competence
    competence = posterior["D"].mean(dim=["chain", "draw"]).values

    # Posterior mean of consensus per question
    z_probs = posterior["Z"].mean(dim=["chain", "draw"]).values
    consensus = np.round(z_probs).astype(int)

    # Compute majority vote across informants
    majority = np.round(data.mean(axis=0)).astype(int)

    # Agreement rate between consensus and majority vote
    agreement = np.mean(consensus == majority) * 100

    return {
        "competence_means": competence,
        "z_means": z_probs,
        "consensus_answers": consensus,
        "majority_vote": majority,
        "agreement_percentage": agreement
    }

def visualize_results(results, analysis):
    """
    Generate plots for model diagnostics and output summaries.
    
    Parameters:
    -----------
    results : dict
        Model outputs (trace, summary).
    analysis : dict
        Analysis outcomes (competence, consensus, etc.).
    
    Returns:
    --------
    dict
        Dictionary of generated plot objects.
    """
    trace = results["trace"]
    plots = {}

    # Diagnostics: trace plots for key variables
    plots["trace_plot"] = az.plot_trace(trace, var_names=["D", "Z"])

    # Posterior distributions
    plots["competence_posterior"] = az.plot_posterior(trace, var_names=["D"])
    plots["z_posterior"] = az.plot_posterior(trace, var_names=["Z"])

    # Competence with credible intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    competence_plot_data = []
    for i, mean in enumerate(analysis["competence_means"]):
        samples = trace.posterior["D"][:, :, i].values.flatten()
        competence_plot_data.append({
            "informant": f"P{i+1}",
            "mean": samples.mean(),
            "lower": np.percentile(samples, 2.5),
            "upper": np.percentile(samples, 97.5)
        })

    # Sort by mean competence
    competence_plot_data.sort(key=lambda d: d["mean"], reverse=True)

    informants = [d["informant"] for d in competence_plot_data]
    means = [d["mean"] for d in competence_plot_data]
    lower = [d["lower"] for d in competence_plot_data]
    upper = [d["upper"] for d in competence_plot_data]

    ax.errorbar(
        means, informants,
        xerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
        fmt='o', capsize=5, color='blue', ecolor='black'
    )
    ax.set_title("Informant Competence Estimates (95% CI)")
    ax.set_xlabel("Competence")
    ax.set_xlim(0.4, 1.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    plots["competence_ranking"] = fig

    # Consensus vs. Majority vote comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(analysis["consensus_answers"]))
    ax.bar(x - 0.175, analysis["consensus_answers"], width=0.35, label="CCT Consensus")
    ax.bar(x + 0.175, analysis["majority_vote"], width=0.35, label="Majority Vote")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_ylabel("Answer (0/1)")
    ax.set_title("Consensus vs. Majority Vote by Question")
    ax.legend()
    plots["consensus_vs_majority"] = fig

    return plots

def main():
    # Attempt to load dataset from potential paths
    paths = [
        "/home/jovyan/cct-midterm/cct-midterm/data/plant_knowledge.csv",
        "../data/plant_knowledge.csv",
        "../../data/plant_knowledge.csv"
    ]
    for path in paths:
        try:
            data = load_plant_knowledge_data(path)
            break
        except FileNotFoundError:
            continue

    # Run the model and analyze results
    results = run_cct_model(data)
    analysis = analyze_results(results, data)
    plots = visualize_results(results, analysis)

    # Print high-level analysis
    print("\n===== CULTURAL CONSENSUS THEORY REPORT =====\n")
    r_hat_max = np.nanmax(results["summary"]["r_hat"].values)
    print(f"Convergence (max R-hat): {r_hat_max:.3f}")
    print("Converged!" if r_hat_max < 1.05 else "Potential convergence issues.")

    # Display sorted competence table
    competence_df = pd.DataFrame({
        "Informant": [f"P{i+1}" for i in range(len(analysis["competence_means"]))],
        "Competence": analysis["competence_means"]
    }).sort_values("Competence", ascending=False)
    print("\nInformant Competence Rankings:\n", competence_df)

    # Top and bottom informants
    top = competence_df.iloc[0]
    bottom = competence_df.iloc[-1]
    print(f"\nMost competent: {top['Informant']} (D = {top['Competence']:.3f})")
    print(f"Least competent: {bottom['Informant']} (D = {bottom['Competence']:.3f})")

    # Consensus vs majority table
    consensus_df = pd.DataFrame({
        "Question": [f"PQ{i+1}" for i in range(len(analysis["consensus_answers"]))],
        "Consensus": analysis["consensus_answers"],
        "Posterior Probability": analysis["z_means"],
        "Majority Vote": analysis["majority_vote"]
    })
    print("\nConsensus Inference:\n", consensus_df)

    print(f"\nConsensus-Majority Agreement: {analysis['agreement_percentage']:.1f}%")
    disagreements = np.where(analysis["consensus_answers"] != analysis["majority_vote"])[0]
    if len(disagreements) > 0:
        print("Disagreements on:", ", ".join([f"PQ{i+1}" for i in disagreements]))
    else:
        print("Full agreement between CCT and majority vote.")

    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
