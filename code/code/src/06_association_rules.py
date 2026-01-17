# src/06_association_rules.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from config import FEATURES_PATH

def main():
    df = pd.read_csv(FEATURES_PATH)

    # biến skills -> dạng list
    transactions = df["skills_str"].fillna("").apply(lambda x: [s for s in x.split(",") if s])

    # one-hot encoding
    all_skills = sorted({s for lst in transactions for s in lst})
    onehot = pd.DataFrame(0, index=df.index, columns=all_skills)

    for idx, lst in enumerate(transactions):
        for s in lst:
            onehot.loc[idx, s] = 1

    # Apriori
    freq = apriori(onehot, min_support=0.03, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(by="lift", ascending=False)

    # lưu kết quả
    rules.to_csv("data/processed/topcv_skill_rules.csv", index=False, encoding="utf-8-sig")
    print("✅ Saved rules: data/processed/topcv_skill_rules.csv")
    print(rules.head(10)[["antecedents", "consequents", "support", "confidence", "lift"]])

if __name__ == "__main__":
    main()
