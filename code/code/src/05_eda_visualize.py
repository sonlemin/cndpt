# src/05_eda_visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURES_PATH, FIG_DIR

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)

    # 1) Top vị trí tuyển dụng phổ biến (theo job_group)
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x="job_group", order=df["job_group"].value_counts().index)
    plt.title("Phân bố nhóm vị trí (job_group)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_job_group_count.png", dpi=200)
    plt.close()

    # 2) Top kỹ năng phổ biến
    skill_series = df["skills_str"].dropna().str.split(",").explode()
    top_skills = skill_series.value_counts().head(20)

    plt.figure(figsize=(10,6))
    top_skills.sort_values().plot(kind="barh")
    plt.title("Top 20 kỹ năng phổ biến")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_top_skills.png", dpi=200)
    plt.close()

   # 3) % tin có lương / không có lương (luôn có đủ 2 nhóm 0 và 1)
    salary_counts = df["has_salary"].value_counts().reindex([0, 1], fill_value=0)

    plt.figure(figsize=(5,5))
    plt.pie(
        salary_counts,
        labels=["Không có lương", "Có lương"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Tỷ lệ tin có công khai lương")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_salary_pie.png", dpi=200)
    plt.close()

    # 4) Phân bố lương (chỉ với tin có lương)
    df_salary = df[df["salary_avg"].notna()].copy()

    if len(df_salary) > 10:
        plt.figure(figsize=(8,5))
        sns.histplot(df_salary["salary_avg"], bins=20)
        plt.title("Phân bố lương trung bình (triệu/tháng)")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/04_salary_hist.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8,5))
        sns.boxplot(data=df_salary, x="job_group", y="salary_avg")
        plt.title("Lương theo nhóm vị trí")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/05_salary_by_group.png", dpi=200)
        plt.close()

    print("✅ Đã xuất biểu đồ vào:", FIG_DIR)

if __name__ == "__main__":
    main()
