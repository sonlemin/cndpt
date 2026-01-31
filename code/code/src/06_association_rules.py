#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

try:
    from config import (
        FEATURES_PATH,
        RULES_PATH,
        ASSOCIATION_REPORT_PATH,
        MIN_SUPPORT,
        MIN_CONFIDENCE,
        MIN_LIFT,
        MAX_RULES,
        FIG_DIR,
    )
except ImportError:
    FEATURES_PATH = "data/processed/topcv_it_features.csv"
    RULES_PATH = "data/processed/topcv_skill_rules.csv"
    ASSOCIATION_REPORT_PATH = "data/processed/association_rules_report.txt"
    MIN_SUPPORT = 0.03
    MIN_CONFIDENCE = 0.3
    MIN_LIFT = 1.0
    MAX_RULES = 100
    FIG_DIR = "reports/figures"


def load_data(filepath):
    print(f"Dang doc du lieu: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Khong tim thay file: {filepath}\n"
            "Vui long chay buoc trich xuat dac trung (04_extract_features.py) truoc."
        )

    print(f"Da doc xong: {df.shape[0]} dong, {df.shape[1]} cot")

    if "skills_str" not in df.columns and "skills" not in df.columns:
        raise ValueError("Khong co cot skills_str hoac skills trong du lieu")

    return df


def prepare_transactions(df):
    print("Dang chuan bi du lieu ky nang")

    if "skills_str" in df.columns:
        transactions = df["skills_str"].fillna("").astype(str).apply(
            lambda x: [s.strip() for s in x.split(",") if s.strip()]
        )
    else:
        transactions = df["skills"].fillna("").astype(str).apply(
            lambda x: [
                s.strip()
                for s in x.replace("[", "").replace("]", "").replace("'", "").split(",")
                if s.strip()
            ]
        )

    total = len(transactions)
    n_with_skills = int((transactions.apply(len) > 0).sum())
    avg_skills = float(transactions.apply(len).mean())

    print(f"So tin co ky nang: {n_with_skills}/{total} ({n_with_skills/total*100:.1f}%)")
    print(f"So ky nang trung binh moi tin: {avg_skills:.2f}")

    transactions2 = transactions[transactions.apply(len) >= 2].tolist()
    print(
        f"So tin co tu 2 ky nang tro len: {len(transactions2)}/{total} ({len(transactions2)/total*100:.1f}%)"
    )

    if len(transactions2) < 10:
        raise ValueError("Qua it tin co tu 2 ky nang tro len de khai thac luat ket hop")

    return transactions2


def create_onehot_matrix(transactions):
    print("Dang tao ma tran one-hot")

    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(arr, columns=te.columns_)

    print(f"So ky nang khac nhau: {onehot.shape[1]}")
    print(f"Kich thuoc ma tran: {onehot.shape[0]} x {onehot.shape[1]}")

    if onehot.shape[1] == 0:
        raise ValueError("Khong co ky nang nao trong du lieu")

    return onehot


def mine_rules(onehot, min_support, min_confidence, min_lift):
    print("Dang khai thac luat ket hop")
    print(f"Tham so: min_support={min_support} ({min_support*100:.1f}%), "
          f"min_confidence={min_confidence} ({min_confidence*100:.1f}%), "
          f"min_lift={min_lift:.2f}")

    freq_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    print(f"So tap pho bien tim duoc: {len(freq_itemsets)}")

    if len(freq_itemsets) == 0:
        return None

    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"So luat sinh ra ban dau: {len(rules)}")

    if len(rules) == 0:
        return None

    rules = rules[rules["lift"] >= min_lift].copy()
    print(f"So luat sau khi loc lift: {len(rules)}")

    if len(rules) == 0:
        return None

    rules = rules[
        (rules["antecedents"].apply(len) <= 2) &
        (rules["consequents"].apply(len) == 1)
    ].copy()
    print(f"So luat sau khi loc do dai (ve trai <=2, ve phai =1): {len(rules)}")

    if len(rules) == 0:
        return None

    rules = rules.sort_values(by="lift", ascending=False)
    return rules


def save_results(rules, rules_path, report_path, max_rules):
    print("Dang luu ket qua")

    if rules is None or len(rules) == 0:
        print("Khong co luat nao de luu")
        return

    os.makedirs(os.path.dirname(rules_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    rules.to_csv(rules_path, index=False, encoding="utf-8-sig")
    print("Da luu file CSV:", rules_path)

    n_show = min(max_rules, len(rules))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BAO CAO LUAT KET HOP (ASSOCIATION RULES)\n")
        f.write("=" * 80 + "\n\n")

        f.write("THAM SO:\n")
        f.write(f"  min_support:    {MIN_SUPPORT} ({MIN_SUPPORT*100:.1f}%)\n")
        f.write(f"  min_confidence: {MIN_CONFIDENCE} ({MIN_CONFIDENCE*100:.1f}%)\n")
        f.write(f"  min_lift:       {MIN_LIFT:.2f}\n")
        f.write(f"  max_rules:      {MAX_RULES}\n\n")

        f.write("TOM TAT:\n")
        f.write(f"  Tong so luat: {len(rules)}\n")
        f.write(f"  Hien thi top: {n_show}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP LUAT (sap xep theo LIFT giam dan)\n")
        f.write("=" * 80 + "\n\n")

        for i, (_, row) in enumerate(rules.head(n_show).iterrows(), start=1):
            antecedents = ", ".join(sorted(row["antecedents"]))
            consequents = ", ".join(sorted(row["consequents"]))
            f.write(f"Luat #{i}:\n")
            f.write(f"  NEU:   {antecedents}\n")
            f.write(f"  THI:   {consequents}\n")
            f.write(f"  Support:    {row['support']:.4f} ({row['support']*100:.2f}%)\n")
            f.write(f"  Confidence: {row['confidence']:.4f} ({row['confidence']*100:.2f}%)\n")
            f.write(f"  Lift:       {row['lift']:.4f}\n\n")

    print("Da luu bao cao:", report_path)


def hien_thi_top_rules(rules, n=10):
    if rules is None or len(rules) == 0:
        return

    n = min(n, len(rules))
    print("=" * 80)
    print(f"TOP {n} LUAT KET HOP")
    print("=" * 80)

    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    top = rules.head(n)[cols].copy()
    top["antecedents"] = top["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    top["consequents"] = top["consequents"].apply(lambda x: ", ".join(sorted(x)))
    top["support"] = top["support"].apply(lambda x: f"{x:.4f}")
    top["confidence"] = top["confidence"].apply(lambda x: f"{x:.4f}")
    top["lift"] = top["lift"].apply(lambda x: f"{x:.2f}")

    print(top.to_string(index=False))


def ve_bieu_do_top_luat_theo_lift(rules, out_dir, top_n=20):
    if rules is None or len(rules) == 0:
        print("Khong co luat de ve bieu do")
        return

    os.makedirs(out_dir, exist_ok=True)

    top = rules.sort_values("lift", ascending=False).head(top_n).copy()

    def rule_to_text(row):
        left = ", ".join(sorted(list(row["antecedents"])))
        right = ", ".join(sorted(list(row["consequents"])))
        return f"{left} -> {right}"

    top["rule_text"] = top.apply(rule_to_text, axis=1)
    top = top.sort_values("lift", ascending=True)

    plt.figure(figsize=(12, max(6, 0.45 * len(top))))
    plt.barh(top["rule_text"], top["lift"])

    for i, v in enumerate(top["lift"].values):
        plt.text(float(v) + 0.02, i, f"{float(v):.2f}", va="center", fontweight="bold")

    plt.title(f"Top {len(top)} luat ket hop theo lift")
    plt.xlabel("Lift")
    plt.ylabel("Luat")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "09_top_rules_by_lift.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Da luu bieu do:", out_path)


def main():
    df = load_data(FEATURES_PATH)
    transactions = prepare_transactions(df)
    onehot = create_onehot_matrix(transactions)

    rules = mine_rules(onehot, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT)

    save_results(rules, RULES_PATH, ASSOCIATION_REPORT_PATH, MAX_RULES)
    hien_thi_top_rules(rules, n=10)
    ve_bieu_do_top_luat_theo_lift(rules, FIG_DIR, top_n=20)

    if rules is None or len(rules) == 0:
        print("Khong tim thay luat nao. Goi y:")
        print(f"  - Giam min_support (hien tai {MIN_SUPPORT})")
        print(f"  - Giam min_confidence (hien tai {MIN_CONFIDENCE})")
        print(f"  - Giam min_lift (hien tai {MIN_LIFT})")
    else:
        print("Hoan tat khai thac luat ket hop")
        print(f"Tong so luat: {len(rules)}")
        print("CSV:", RULES_PATH)
        print("Bao cao:", ASSOCIATION_REPORT_PATH)
        print("Thu muc bieu do:", FIG_DIR)


if __name__ == "__main__":
    main()
