"""
Spatial Diffusion - Python translation of the R script
Original R script: Spatial_diffusion_NE_25March2026_annotated.R

Notes
-----
- This script mirrors the logic of the R code as closely as possible.
- You may need to adapt file paths, separators, and package availability on your machine.
- Some outputs depend on the exact structure of the CSV files in the original project folder.
- A few R graphics idioms were translated into standard matplotlib equivalents.
- The original R script contains one potentially unusual formula in the country-level Bass block:
  the numerator of `Bcdf2` uses `ngete` while the denominator uses `ngete2`.
  This translation preserves that behavior, but you may want to verify it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import networkx as nx


# =============================================================================
# Housekeeping
# =============================================================================

# Set the working directory to the folder containing the input data files and
# where the output figures will be saved.
# NOTE: This path is machine-specific and may need to be changed on another computer.
WORKDIR = Path(r"c:\Users\lengyel.balazs\Desktop\LengyelB_2022June\Northeastern teaching\coding class")
os.chdir(WORKDIR)


# =============================================================================
# 1. Scaling and gravity
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 Scaling over the life cycle
# -----------------------------------------------------------------------------

# Read individual-level adoption category data.
id_adopter = pd.read_csv("id_adopter.csv", sep=",")
id_cityid = pd.read_csv("id_cityid.csv", sep=",")
pop = pd.read_csv("cityid_pop_poplog_2557.csv", sep=";")

# Merge user-to-city data with adopter category data.
id_city_adopt = pd.merge(id_cityid, id_adopter, on="id", how="inner")
id_city_adopt["n"] = 1

# Collapse adopter categories into 3 broader life-cycle stages.
id_city_adopt["ad3"] = 1
id_city_adopt.loc[id_city_adopt["adopter"] == 2, "ad3"] = 2
id_city_adopt.loc[id_city_adopt["adopter"].isin([3, 4, 5]), "ad3"] = 3

# Count adopters by city and stage.
city_adopt = (
    id_city_adopt.groupby(["cityid", "ad3"], as_index=False)["n"]
    .sum()
    .rename(columns={"n": "N"})
)
city_adopt = pd.merge(city_adopt, pop, on="cityid", how="left")

# Restrict to cities above 10,000 inhabitants and log-transform number of adopters.
c_a = city_adopt.loc[city_adopt["pop"] > 10000].copy()
c_a["N_log"] = np.log10(c_a["N"])


def fit_ols(df, y_col, x_col):
    X = sm.add_constant(df[x_col])
    return sm.OLS(df[y_col], X, missing="drop").fit()


ad1_df = c_a.loc[c_a["ad3"] == 1].copy()
ad2_df = c_a.loc[c_a["ad3"] == 2].copy()
ad3_df = c_a.loc[c_a["ad3"] == 3].copy()

ad1 = fit_ols(ad1_df, "N_log", "pop_log")
ad2 = fit_ols(ad2_df, "N_log", "pop_log")
ad3 = fit_ols(ad3_df, "N_log", "pop_log")

print(ad1.summary())
print(ad2.summary())
print(ad3.summary())


def add_predictions(df, model, x_col):
    X = sm.add_constant(df[x_col])
    pred = model.get_prediction(X).summary_frame(alpha=0.05)
    out = df.copy()
    out["fit"] = pred["mean"].values
    out["lwr"] = pred["mean_ci_lower"].values
    out["upr"] = pred["mean_ci_upper"].values
    return out.sort_values(x_col)


plot_1 = add_predictions(ad1_df, ad1, "pop_log")
plot_2 = add_predictions(ad2_df, ad2, "pop_log")
plot_3 = add_predictions(ad3_df, ad3, "pop_log")

fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
ax.scatter(plot_1["pop_log"], plot_1["N_log"], color="red", marker="^", s=40, label="Innovators")
ax.plot(plot_1["pop_log"], plot_1["fit"], color="red")
ax.fill_between(plot_1["pop_log"], plot_1["lwr"], plot_1["upr"], color="red", alpha=0.3)

ax.scatter(plot_2["pop_log"], plot_2["N_log"], color="green", marker="+", s=60, label="Early adopters")
ax.plot(plot_2["pop_log"], plot_2["fit"], color="green")
ax.fill_between(plot_2["pop_log"], plot_2["lwr"], plot_2["upr"], color="green", alpha=0.3)

ax.scatter(plot_3["pop_log"], plot_3["N_log"], color="blue", marker="x", s=40, label="Majority & Laggards")
ax.plot(plot_3["pop_log"], plot_3["fit"], color="blue")
ax.fill_between(plot_3["pop_log"], plot_3["lwr"], plot_3["upr"], color="blue", alpha=0.3)

ax.set_xlim(4, 6.5)
ax.set_ylim(0, 6)
ax.legend(frameon=False, loc="lower left")
ax.set_xticks([4, 4.5, 5, 5.5, 6, 6.5])
ax.set_xticklabels([r"$10^{4}$", r"$10^{4.5}$", r"$10^{5}$", r"$10^{5.5}$", r"$10^{6}$", r"$10^{6.5}$"])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels([r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"])
ax.text(5.8, 0.65, r"$\beta=1.41$, SE=0.09", color="red", fontsize=12)
ax.text(5.8, 0.35, r"$\beta=1.28$, SE=0.04", color="green", fontsize=12)
ax.text(5.8, 0.05, r"$\beta=1.07$, SE=0.01", color="blue", fontsize=12)
fig.savefig("fig1_scaling.png", bbox_inches="tight")
plt.close(fig)


# -----------------------------------------------------------------------------
# 1.2 Gravity over the life cycle
# -----------------------------------------------------------------------------

inv_grav = pd.read_csv("inv_gravity.csv", sep=",")
print("R² innovators:", inv_grav[["P_d_1_log", "group_gravity_log"]].corr().iloc[0, 1] ** 2)
print("R² early adopters:", inv_grav[["P_d_2_log", "group_gravity_log"]].corr().iloc[0, 1] ** 2)
print("R² majority/laggards:", inv_grav[["P_d_345_log", "group_gravity_log"]].corr().iloc[0, 1] ** 2)

mask = (inv_grav["group_gravity"] > 5) & (inv_grav["group_gravity"] < 305)
ig = inv_grav.loc[mask].copy()

fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
ax.plot(ig["group_gravity_log"], ig["P_d_1_log"], "o-", color="red", label="Innovators")
ax.plot(ig["group_gravity_log"], ig["P_d_2_log"], "o-", color="green", label="Early adopters")
ax.plot(ig["group_gravity_log"], ig["P_d_345_log"], "o-", color="blue", label="Majority & laggards")
ax.plot(ig["group_gravity_log"], ig["line1"], linestyle=":", color="black", linewidth=2)
ax.plot(ig["group_gravity_log"], ig["line2"], linestyle="--", color="black", linewidth=2)
ax.plot(ig["group_gravity_log"], ig["line3"], linestyle="-", color="black", linewidth=2)
ax.set_xlim(1, 2.5)
ax.set_ylim(-7.8, -4.8)
ax.set_xlabel("Distance")
ax.set_ylabel("Probability of Invitations")
ax.set_xticks([1, 1.5, 2, 2.5])
ax.set_xticklabels([r"$10^{1}$", r"$10^{1.5}$", r"$10^{2}$", r"$10^{2.5}$"])
ax.set_yticks([-8, -7, -6, -5])
ax.set_yticklabels([r"$10^{-8}$", r"$10^{-7}$", r"$10^{-6}$", r"$10^{-5}$"])
ax.legend(frameon=False, loc="lower left")
fig.savefig("fig1_inv_grav.png", bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 2. Bass DE estimation
# =============================================================================

c = pd.read_csv("BASS_curve_country.csv", sep=";")
ctime = c["month"].to_numpy(dtype=float)

P = 0.000223
Q = 0.094
ngete = np.exp(-(P + Q) * ctime)
Bcdf = ((1 - ngete) / (1 + (Q / P) * ngete))
creg = c["reg_rate_month"].to_numpy(dtype=float)


def bass_density(time, cP, cQ, M=0.31311578):
    exp_term = np.exp(-(cP + cQ) * time)
    return M * (((cP + cQ) ** 2 / cP) * exp_term / (1 + (cQ / cP) * exp_term) ** 2)


params_country, _ = curve_fit(
    lambda t, cP, cQ: bass_density(t, cP, cQ, M=0.31311578),
    ctime,
    creg,
    p0=[0.00696, 0.0964],
    maxfev=100000,
)
cp, cq = params_country
c["smooth"] = c["reg_month_country"].rolling(window=5).mean()

Bcdf_m = np.empty_like(Bcdf)
Bcdf_m[0] = Bcdf[0]
Bcdf_m[1:] = np.diff(Bcdf)

# Preserve the original R code exactly here.
p, q = cp, cq
ngete2 = np.exp(-(p + q) * ctime)
Bcdf2 = ((1 - ngete) / (1 + (q / p) * ngete2))
Bcdf_m2 = np.empty_like(Bcdf2)
Bcdf_m2[0] = Bcdf2[0]
Bcdf_m2[1:] = np.diff(Bcdf2)

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.plot(ctime, c["reg_month_country"], "o-", color="orange", linewidth=2, label="New users")
valid = c["smooth"].notna()
ax.plot(c.loc[valid, "month"], c.loc[valid, "smooth"], color="blue", linewidth=2, label="Smoothed Adoption")
ax.plot(ctime, Bcdf_m2 * c["reg_month_country"].sum(), color="red", linewidth=3, label="Estimated Adoption")
est_peak_x = ctime[np.argmax(Bcdf_m2)]
smoothed_months = c.loc[valid, "month"].to_numpy()
smoothed_peak_x = smoothed_months[np.argmax(c.loc[valid, "smooth"].to_numpy())]
ax.axvline(est_peak_x, color="red", linestyle="--", linewidth=3)
ax.axvline(smoothed_peak_x, color="blue", linestyle=":", linewidth=3)
ax.legend(frameon=False, loc="upper right")
ax.set_xlabel("months")
ax.annotate("", xy=(10, 85000), xytext=(20, 85000), arrowprops=dict(arrowstyle="<->", color="black", lw=2))
ax.text(15, 70000, "Prediction Error", ha="center", fontsize=12)
ax.text(26, 100000, "Est.\nPeak", ha="center", fontsize=12)
ax.text(4, 100000, "Smo.\nPeak", ha="center", fontsize=12)
fig.savefig("fig2_Bass_global_Balazs.png", bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 3. Estimate Bass DE by towns
# =============================================================================

e = pd.read_csv("BASS_curve_towns_2557.csv", sep=";")
cities = pd.read_csv("cityid_pop_poplog_2557.csv", sep=";")
n_cities = len(cities)

qa = np.full(n_cities, np.nan)
pa = np.full(n_cities, np.nan)
qd = np.full(n_cities, np.nan)
xd = np.full(n_cities, np.nan)
SSa = np.full(n_cities, np.nan)
SSd = np.full(n_cities, np.nan)
mt = np.full(n_cities, np.nan)


def disad_cum_func(time, x, q):
    return x * (1 + q) ** time


def bass_density_fixed_m(time, P, Q, m):
    exp_term = np.exp(-(P + Q) * time)
    return m * (((P + Q) ** 2 / P) * exp_term / (1 + (Q / P) * exp_term) ** 2)


for i in range(1, 2558):
    town = e.loc[e["cityid_new"] == i].copy()
    if town.empty:
        continue
    time = town["month"].to_numpy(dtype=float)
    disad = town["disadoption_rate_month"].to_numpy(dtype=float)
    cumdisad = np.cumsum(disad)
    try:
        params_d, _ = curve_fit(disad_cum_func, time, cumdisad, p0=[1e-6, 0.1], maxfev=100000)
        xd[i - 1], qd[i - 1] = params_d
        SSd[i - 1] = np.sum((cumdisad - disad_cum_func(time, *params_d)) ** 2)
    except Exception:
        continue

for i in range(1, 2558):
    town = e.loc[e["cityid_new"] == i].copy()
    if town.empty:
        continue
    time = town["month"].to_numpy(dtype=float)
    adopt = town["adoption_rate_month"].to_numpy(dtype=float)
    cumadopt = np.cumsum(adopt)
    m = np.max(cumadopt)
    try:
        params_a, _ = curve_fit(lambda t, P, Q: bass_density_fixed_m(t, P, Q, m), time, adopt, p0=[0.0000696, 0.1], maxfev=100000)
        pa[i - 1], qa[i - 1] = params_a
        SSa[i - 1] = np.sum((adopt - bass_density_fixed_m(time, *params_a, m)) ** 2)
        mt[i - 1] = m
    except Exception:
        continue

T_emp3 = np.full(n_cities, np.nan)
for i in range(1, 2558):
    smoothing = e.loc[e["cityid_new"] == i].copy()
    if smoothing.empty:
        continue
    smoothing["peak"] = smoothing["adoption_rate_month"].rolling(window=3).mean()
    peak_val = smoothing["peak"].max(skipna=True)
    peak_rows = smoothing.loc[smoothing["peak"] == peak_val]
    if len(peak_rows) >= 3:
        T_emp3[i - 1] = peak_rows["month"].iloc[2]
    elif len(peak_rows) > 0:
        T_emp3[i - 1] = peak_rows["month"].iloc[-1]

emp_peak = pd.DataFrame({"cityid": e["cityid"].drop_duplicates().sort_values().to_numpy()[:n_cities], "T_emp3": T_emp3})

data_pop = pd.read_csv("population.csv", sep=",")
data_peak = pd.read_csv("T_adoption_peak.csv", sep=",")
data_pq = pd.read_csv("qa_qd_data.csv", sep=",")
data_code = pd.read_csv("city_codes_pop.csv", sep=",")
data_code = pd.merge(data_code, data_peak, on="cityid", how="inner")
if len(data_code.columns) >= 5:
    data_code = data_code.rename(columns={data_code.columns[4]: "peak"})
data_OSN = pd.merge(data_code, data_pq, on="cityid", how="inner")

for idxs in ([9, 10, 11, 12, 13, 14], [13, 14, 15, 16]):
    cols = [data_OSN.columns[i] for i in idxs if i < len(data_OSN.columns)]
    data_OSN = data_OSN.drop(columns=cols, errors="ignore")

for col in ["cityid", "pop_ext", "ksh_code", "pop"]:
    if col in data_OSN.columns:
        data_OSN[col] = pd.to_numeric(data_OSN[col], errors="coerce")

rename_map = {}
if len(data_OSN.columns) > 10:
    rename_map[data_OSN.columns[10]] = "qa"
if len(data_OSN.columns) > 11:
    rename_map[data_OSN.columns[11]] = "qd"
if len(data_OSN.columns) > 12:
    rename_map[data_OSN.columns[12]] = "xd"
data_OSN = data_OSN.rename(columns=rename_map)
if "pa" not in data_OSN.columns:
    data_OSN["pa"] = pa[:len(data_OSN)]
if "qa" not in data_OSN.columns:
    data_OSN["qa"] = qa[:len(data_OSN)]

data_OSN["Tpq"] = -np.log(data_OSN["pa"] / data_OSN["qa"]) / (data_OSN["pa"] + data_OSN["qa"])
if "cityid" in data_OSN.columns:
    data_OSN = pd.merge(data_OSN, emp_peak, on="cityid", how="left")

dens_df = data_OSN[["Tpq", "T_emp3"]].dropna().copy()
x = dens_df["Tpq"].to_numpy()
y = dens_df["T_emp3"].to_numpy()
z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y])) if len(dens_df) > 10 else np.ones_like(x)
idx = np.argsort(z)
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots(figsize=(7.5, 8.75), dpi=100)
sc = ax.scatter(x, y, c=z, s=20, cmap="turbo")
ax.plot([40, 80], [40, 80], color="white", linewidth=3)
ax.set_xlim(40, 80)
ax.set_ylim(40, 80)
ax.set_xlabel("Estimated peak (month)")
ax.set_ylabel("Empirical peak (month)")
plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, label="Kernel density")
fig.savefig("TBass_Temp_smooth_b.png", bbox_inches="tight")
plt.close(fig)

data_OSN["T1"] = -np.log(0.5e-4 / data_OSN["qa"]) / (0.5e-4 + data_OSN["qa"])
data_OSN["T2"] = -np.log(1e-4 / data_OSN["qa"]) / (1e-4 + data_OSN["qa"])
data_OSN["T3"] = -np.log(3e-4 / data_OSN["qa"]) / (3e-4 + data_OSN["qa"])
data_OSN["T4"] = -np.log(5e-4 / data_OSN["qa"]) / (5e-4 + data_OSN["qa"])

p_q_plot = pd.DataFrame({
    "cities.cityid": cities["cityid"],
    "cityid_new": cities["cityid_new"] if "cityid_new" in cities.columns else np.arange(1, len(cities) + 1),
    "pa": pa,
    "qa": qa,
})
if "cityid" in data_OSN.columns:
    p_q_plot = pd.merge(p_q_plot, data_OSN, left_on="cities.cityid", right_on="cityid", how="inner")
if "cityid_new" in p_q_plot.columns:
    p_q_plot = p_q_plot.loc[p_q_plot["cityid_new"] < 1001].copy()

fig, ax = plt.subplots(figsize=(7.5, 8.75), dpi=100)
scatter = ax.scatter(p_q_plot["qa"], p_q_plot["Tpq"], c=p_q_plot["pa"], cmap="rainbow", s=40)
ax.plot(data_OSN["qa"], data_OSN["T1"], color="red")
ax.plot(data_OSN["qa"], data_OSN["T2"], color="orange")
ax.plot(data_OSN["qa"], data_OSN["T3"], color="#1affcc")
ax.plot(data_OSN["qa"], data_OSN["T4"], color="blue")
ax.set_xlim(0.05, 0.3)
ax.set_ylim(45, 95)
ax.set_xlabel("Q adoption")
ax.set_ylabel("Peak month")
plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.1, label="P adoption")
ax.text(0.27, 95, "Fixed P", color="black", fontsize=14)
fig.savefig("Tpq_qa_col_p.png", bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 4. Toy example of the BASS ABM
# =============================================================================

ver = pd.read_csv("vertices_sample.csv")
edg = pd.read_csv("edges_sample.csv")

if edg.shape[1] < 2:
    raise ValueError("edges_sample.csv must contain at least two columns (source, target).")
source_col = edg.columns[0]
target_col = edg.columns[1]
network_fs = nx.from_pandas_edgelist(edg, source=source_col, target=target_col)

if "month" not in ver.columns:
    raise ValueError("vertices_sample.csv must contain a 'month' column.")
node_id_col = ver.columns[0]
node_ids = ver[node_id_col].tolist()
diffusers_fs = set(ver.loc[ver["month"] == 3, node_id_col].tolist())
susceptibles_fs = set(node_ids) - diffusers_fs

time_fs = {node: 0 for node in node_ids}
for node in diffusers_fs:
    time_fs[node] = 1

p_fs = 0.000104
q_fs = 0.12
neigh = {node: set([node]) | set(network_fs.neighbors(node)) for node in node_ids if node in network_fs}

rng = np.random.default_rng(1042)
time_adoption = 2
while time_adoption < 129:
    a = {}
    for node, ego_set in neigh.items():
        denom = max(len(ego_set) - 1, 1)
        a[node] = sum(1 for n in ego_set if n in diffusers_fs) / denom

    adopters = []
    for node in list(susceptibles_fs):
        prob = p_fs + a.get(node, 0) * q_fs
        if rng.random() < prob:
            adopters.append(node)

    for node in adopters:
        time_fs[node] = time_adoption
    diffusers_fs.update(adopters)
    susceptibles_fs = set(node_ids) - diffusers_fs
    neigh = {node: ego for node, ego in neigh.items() if node not in diffusers_fs}
    print(sum(1 for v in time_fs.values() if v != 0), "-->", time_adoption)
    time_adoption += 1

adoption_times = np.array(list(time_fs.values()))
adoption_times = adoption_times[adoption_times > 0].astype(int)
counts = np.bincount(adoption_times)
results = pd.DataFrame({"Time": np.arange(1, len(counts)), "CDF": np.cumsum(counts[1:]) / len(node_ids)})

fig, ax = plt.subplots()
ax.plot(results["Time"], results["CDF"], "o-")
ax.set_ylabel("CDF")
ax.set_xlabel("Time")
fig.savefig("toy_abm_cdf.png", bbox_inches="tight")
plt.close(fig)

# Full network ABM in the original script:
# https://github.com/bokae/spatial_diffusion
