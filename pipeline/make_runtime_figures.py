import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
DATA_DIR = "benchmark_results"
OUT_DIR = "figures_runtime"
os.makedirs(OUT_DIR, exist_ok=True)

EXP_A = os.path.join(DATA_DIR, "performance_log_ExperimentA.csv")
EXP_B = os.path.join(DATA_DIR, "performance_log_ExperimentB.csv")
EXP_C = os.path.join(DATA_DIR, "performance_log_ExperimentC.csv")
EXP_D = os.path.join(DATA_DIR, "performance_log_ExperimentD.csv")
EXP_E = os.path.join(DATA_DIR, "performance_log_ExperimentE.csv")


def save_plot(filename):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# -------------------------
# Experiment A
# Per-module runtime
# -------------------------
df_a = pd.read_csv(EXP_A)

components = [
    "SBD",
    "CLASSIFICATION",
    "LOGO_DETECTION",
    "SELECTION_LOGIC",
    "VIDEO_ASSEMBLY",
]

module_means = (
    df_a[df_a["component"].isin(components)]
    .groupby("component")["runtime_s"]
    .mean()
    .reindex(components)
)

label_map = {
    "SBD": "SBD",
    "CLASSIFICATION": "Classification",
    "LOGO_DETECTION": "Logo detection",
    "SELECTION_LOGIC": "Selection",
    "VIDEO_ASSEMBLY": "Assembly",
}

plt.figure(figsize=(8, 5))
plt.bar([label_map[c] for c in module_means.index], module_means.values)
plt.ylabel("Mean runtime (s)")
plt.xlabel("Pipeline component")
plt.title("Per-module runtime breakdown")
plt.xticks(rotation=25, ha="right")
save_plot("fig_module_runtime.png")


# -------------------------
# Experiment B
# Time budget effect
# Assumes runs are ordered as 60, 120, 180, 300
# -------------------------
df_b = pd.read_csv(EXP_B)

budget_order = [60, 120, 180, 300]
run_ids = list(df_b["run_id"].drop_duplicates())

budget_map = dict(zip(run_ids, budget_order))
df_b["time_budget_s"] = df_b["run_id"].map(budget_map)

full_b = df_b[df_b["component"] == "FULL_PIPELINE"].copy()
assembly_b = df_b[df_b["component"] == "VIDEO_ASSEMBLY"].copy()

plt.figure(figsize=(7, 5))
plt.plot(full_b["time_budget_s"], full_b["runtime_s"], marker="o", label="Warm generation total")
plt.plot(assembly_b["time_budget_s"], assembly_b["runtime_s"], marker="o", label="Video assembly")
plt.xlabel("Time budget (s)")
plt.ylabel("Runtime (s)")
plt.title("Effect of time budget on runtime")
plt.legend()
save_plot("fig_time_budget_runtime.png")


# -------------------------
# Experiment C
# Cold vs warm execution
# Assumes one cold and one warm run.
# If your notes identify run order differently, adjust labels.
# -------------------------
df_c = pd.read_csv(EXP_C)

full_c = df_c[df_c["component"] == "FULL_PIPELINE"].copy()
full_c = full_c.sort_values("runtime_s", ascending=False)

labels = ["Cold run", "Warm run"]
runtimes = full_c["runtime_s"].values[:2]

plt.figure(figsize=(6, 5))
plt.bar(labels, runtimes)
plt.ylabel("Runtime (s)")
plt.title("Cold vs warm pipeline execution")
save_plot("fig_cold_vs_warm.png")


# -------------------------
# Experiment D
# Input variation
# -------------------------
df_d = pd.read_csv(EXP_D)

full_d = df_d[df_d["component"] == "FULL_PIPELINE"].copy()
input_runtime = (
    full_d.groupby("game_id")["runtime_s"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(7, 5))
plt.bar(input_runtime.index.astype(str), input_runtime.values)
plt.xlabel("Game ID")
plt.ylabel("Runtime (s)")
plt.title("Runtime across different input games")
save_plot("fig_input_variation.png")


# -------------------------
# Experiment E
# Hardware comparison
# -------------------------
df_e = pd.read_csv(EXP_E)

full_e = df_e[df_e["component"] == "FULL_PIPELINE"].copy()

# Gaming PC baseline from Experiment A
gaming_pc_runtime = df_a[df_a["component"] == "FULL_PIPELINE"]["runtime_s"].mean()

# Laptop from Experiment E
laptop_runtime = full_e["runtime_s"].mean()

plt.figure(figsize=(6, 5))
plt.bar(["Gaming PC", "Laptop"], [gaming_pc_runtime, laptop_runtime])
plt.ylabel("Runtime (s)")
plt.title("Hardware comparison")
save_plot("fig_hardware_comparison.png")


print("All figures created.")