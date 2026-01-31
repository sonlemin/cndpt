#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import FEATURES_PATH, FIG_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def create_job_group_chart(df, out_dir):
    if "job_group" not in df.columns:
        print("⚠️  Missing job_group")
        return

    counts = df["job_group"].value_counts()
    colors = ["#d62728" if g == "other" else "#1f77b4" for g in counts.index]

    plt.figure(figsize=(14, 6))
    plt.bar(counts.index, counts.values, color=colors)

    for i, v in enumerate(counts.values):
        plt.text(i, v + 1, str(int(v)), ha="center", va="bottom", fontweight="bold")

    plt.title("Phân bố nhóm vị trí IT")
    plt.xlabel("Job Group")
    plt.ylabel("Số lượng")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_job_group_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    other_pct = counts.get("other", 0) / len(df) * 100
    print(f"📊 other: {counts.get('other', 0)}/{len(df)} ({other_pct:.1f}%)")


def _skill_series(df):
    if "skills_str" in df.columns:
        s = df["skills_str"].dropna().astype(str).str.split(",").explode()
        s = s[s.str.strip() != ""]
        return s

    if "skills" in df.columns:
        s = df["skills"].dropna().astype(str)
        s = s.str.replace(r"[\[\]']", "", regex=True).str.split(",").explode()
        s = s[s.str.strip() != ""]
        return s

    return pd.Series([], dtype=str)


def create_top_skills_chart(df, out_dir, top_n=20):
    s = _skill_series(df)
    if s.empty:
        print("⚠️  No skills data")
        return

    top = s.value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    ax = top.sort_values().plot(kind="barh", color="steelblue")

    pad = max(2, int(0.02 * top.max()))
    for i, (name, val) in enumerate(top.sort_values().items()):
        ax.text(int(val) + pad, i, str(int(val)), va="center", fontweight="bold")

    plt.title(f"Top {top_n} kỹ năng phổ biến")
    plt.xlabel("Số lượng tin tuyển dụng")
    plt.ylabel("Kỹ năng")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_top_skills.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_salary_disclosure_pie(df, out_dir):
    if "has_salary" in df.columns:
        has_salary = df["has_salary"].fillna(0).astype(int)
    elif "salary_avg" in df.columns:
        has_salary = df["salary_avg"].notna().astype(int)
    else:
        print("⚠️  No salary columns")
        return

    counts = has_salary.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    yes = int(counts.get(1, 0))
    no = int(counts.get(0, 0))

    plt.figure(figsize=(8, 8))

    if yes == 0:
        plt.text(
            0.5,
            0.5,
            "⚠️  100% tin không có thông tin lương\n\n"
            "Lưu ý: Nhiều tin ghi 'Thỏa thuận'\n"
            "nên dữ liệu lương bị hạn chế",
            ha="center",
            va="center",
            fontsize=15,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )
        plt.axis("off")
    else:
        labels = [
            f"Không công khai\n{no} tin ({no/total*100:.1f}%)",
            f"Có công khai\n{yes} tin ({yes/total*100:.1f}%)",
        ]
        plt.pie(
            [no, yes],
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#ff9999", "#66b3ff"],
            textprops={"fontsize": 12},
        )
        plt.title("Tỷ lệ công khai lương trong tin tuyển dụng", fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_salary_disclosure_pie.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💰 Salary disclosure: {yes}/{total} ({yes/total*100:.1f}%)")


def create_salary_bucket_chart(df, out_dir):
    if "salary_avg" not in df.columns:
        print("⚠️  Missing salary_avg")
        return

    df_salary = df[df["salary_avg"].notna()].copy()
    if len(df_salary) < 10:
        print("⚠️  Too few salary records for bucket chart")
        return

    bins = [0, 10, 15, 20, 30, 100]
    labels = ["<10", "10–15", "15–20", "20–30", ">30"]
    df_salary["salary_bucket"] = pd.cut(df_salary["salary_avg"], bins=bins, labels=labels)

    c = df_salary["salary_bucket"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(c.index.astype(str), c.values, color="#66b3ff")
    for i, v in enumerate(c.values):
        plt.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontweight="bold")
    plt.title("Phân bố mức lương (triệu VND) - theo khoảng")
    plt.xlabel("Khoảng lương")
    plt.ylabel("Số lượng tin")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/04_salary_bucket.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_salary_by_group(df, out_dir):
    if "salary_avg" not in df.columns or "job_group" not in df.columns:
        print("⚠️  Missing salary_avg/job_group")
        return

    df_salary = df[df["salary_avg"].notna()].copy()
    if df_salary.empty:
        print("⚠️  No salary data")
        return

    group_counts = df_salary["job_group"].value_counts()
    groups = group_counts[group_counts >= 3].index.tolist()
    df_salary = df_salary[df_salary["job_group"].isin(groups)]
    df_salary = df_salary[df_salary["job_group"] != "other"]

    if df_salary.empty:
        print("⚠️  Not enough salary data by group")
        return

    q1 = df_salary["salary_avg"].quantile(0.25)
    q3 = df_salary["salary_avg"].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df_salary = df_salary[(df_salary["salary_avg"] >= low) & (df_salary["salary_avg"] <= high)]

    if len(df_salary) < 5:
        print("⚠️  Salary data too small after outlier removal")
        return

    order = df_salary.groupby("job_group")["salary_avg"].median().sort_values(ascending=False).index

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_salary, x="job_group", y="salary_avg", order=order, palette="Set2", showfliers=False)
    plt.title("Lương trung bình theo nhóm vị trí (đã loại outlier)")
    plt.xlabel("Job Group")
    plt.ylabel("Lương trung bình (triệu VND)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/05_salary_by_group.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_experience_charts(df, out_dir):
    if "experience_years" not in df.columns:
        print("⚠️  Missing experience_years")
        return

    df_exp = df[df["experience_years"].notna()].copy()
    if len(df_exp) < 10:
        print("⚠️  Too few experience records")
        return

    df_exp = df_exp[df_exp["experience_years"] <= 10].copy()

    plt.figure(figsize=(12, 6))
    bins = np.arange(0, 11, 0.5)
    plt.hist(df_exp["experience_years"], bins=bins, edgecolor="black", alpha=0.7)

    median_val = df_exp["experience_years"].median()
    mean_val = df_exp["experience_years"].mean()
    plt.axvline(median_val, linestyle="--", linewidth=2, label=f"Median: {median_val:.1f}")
    plt.axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}")

    plt.title("Phân bố yêu cầu kinh nghiệm (0–10 năm)")
    plt.xlabel("Số năm kinh nghiệm")
    plt.ylabel("Số lượng")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_experience_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "job_group" not in df_exp.columns:
        return

    group_counts = df_exp["job_group"].value_counts()
    groups = group_counts[group_counts >= 5].index.tolist()
    df_plot = df_exp[df_exp["job_group"].isin(groups)].copy()

    if df_plot.empty:
        print("⚠️  Not enough experience data by group")
        return

    q1 = df_plot["experience_years"].quantile(0.25)
    q3 = df_plot["experience_years"].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df_plot = df_plot[(df_plot["experience_years"] >= low) & (df_plot["experience_years"] <= high)]

    if len(df_plot) < 10:
        print("⚠️  Experience data too small after outlier removal")
        return

    order = df_plot.groupby("job_group")["experience_years"].median().sort_values(ascending=False).index

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_plot, x="job_group", y="experience_years", order=order, palette="Set2", showfliers=False)
    plt.title("Kinh nghiệm theo nhóm vị trí (0–10 năm, đã loại outlier)")
    plt.xlabel("Job Group")
    plt.ylabel("Kinh nghiệm (năm)")
    plt.ylim(0, 10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_experience_by_group.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_correlation_matrix(df, out_dir):
    cols = []
    if "experience_years" in df.columns:
        cols.append("experience_years")
    if "n_skills" in df.columns:
        cols.append("n_skills")
    if "salary_avg" in df.columns and df["salary_avg"].notna().sum() >= 10:
        cols.append("salary_avg")

    if len(cols) < 2:
        print("⚠️  Not enough numeric columns for correlation")
        return

    df_num = df[cols].dropna()
    if len(df_num) < 10:
        print("⚠️  Not enough rows for correlation")
        return

    plt.figure(figsize=(8, 7))
    corr = df_num.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/08_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    print("✅ Loaded:", df.shape)

    create_job_group_chart(df, FIG_DIR)
    create_top_skills_chart(df, FIG_DIR, top_n=20)
    create_salary_disclosure_pie(df, FIG_DIR)
    create_salary_bucket_chart(df, FIG_DIR)
    create_salary_by_group(df, FIG_DIR)
    create_experience_charts(df, FIG_DIR)
    create_correlation_matrix(df, FIG_DIR)

    print("✅ EDA completed. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
