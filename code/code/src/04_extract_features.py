import re
import pandas as pd
from collections import Counter

from config import CLEAN_PATH, FEATURES_PATH, SKILL_PATTERNS, JOB_GROUP_RULES


def extract_real_content(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""

    raw = str(text)
    t = raw.lower()

    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\s+", " ", t)

    markers = [
        "topcv - tiếp lợi thế",
        "chọn đúng việc đi đúng hướng",
        "tải app topcv",
        "ứng tuyển ngay",
        "việc làm mới nhất",
        "việc làm hot",
        "tìm việc làm",
        "cẩm nang nghề nghiệp",
        "chính sách bảo mật",
        "điều khoản dịch vụ",
    ]
    cut_positions = [t.find(m) for m in markers if t.find(m) != -1]

    spam_match = re.search(r"(?:\bviệc\s+làm\b.{0,40}){15,}", t)
    if spam_match:
        cut_positions.append(spam_match.start())

    cut_at = min(cut_positions) if cut_positions else None
    if cut_at is not None and cut_at > 0:
        t = t[:cut_at]

    t = re.sub(
        r"tuyển\s+lập\s+trình\s+viên\s+php\s+tuyển\s+lập\s+trình\s+viên\s+java\s+tuyển\s+lập\s+trình\s+viên\s+(?:__)?dotnet(?:__)?",
        " ",
        t,
        flags=re.IGNORECASE,
    )

    if len(t) < 800 and len(raw) > 5000:
        tl = raw.lower()
        anchors = [
            "mô tả công việc",
            "job description",
            "yêu cầu ứng viên",
            "yêu cầu công việc",
            "quyền lợi",
            "benefits",
            "trách nhiệm",
            "requirements",
        ]
        starts = [tl.find(a) for a in anchors if tl.find(a) != -1]
        if starts:
            s = min([p for p in starts if p >= 0])
            t2 = tl[s : s + 25000]
            t2 = re.sub(r"https?://\S+|www\.\S+", " ", t2)
            t2 = re.sub(r"\s+", " ", t2)
            if len(t2) > len(t):
                t = t2

    return t.strip()


def detect_job_group(title_clean: str) -> str:
    for group, pattern in JOB_GROUP_RULES:
        if re.search(pattern, title_clean, re.IGNORECASE):
            return group
    return "other"


def extract_salary(title: str, text: str):
    title = "" if pd.isna(title) else str(title)
    text = "" if pd.isna(text) else str(text)
    combined = (title + " " + extract_real_content(text))[:6000].lower()

    if "thỏa thuận" in combined or "thoả thuận" in combined:
        return (None, None, None)

    usd_range = re.search(r"(\d+(?:[\.,]\d+)?)\s*[-~]+\s*(\d+(?:[\.,]\d+)?)\s*usd", combined)
    if usd_range:
        a = float(usd_range.group(1).replace(",", "."))
        b = float(usd_range.group(2).replace(",", "."))
        return (a, b, (a + b) / 2)

    usd_single = re.search(r"(\d+(?:[\.,]\d+)?)\s*usd", combined)
    if usd_single:
        val = float(usd_single.group(1).replace(",", "."))
        return (val, val, val)

    vnd_range = re.search(r"(\d+(?:[\.,]\d+)?)\s*[-~đến]+\s*(\d+(?:[\.,]\d+)?)\s*(?:triệu|tr)\b", combined)
    if vnd_range:
        a = float(vnd_range.group(1).replace(",", "."))
        b = float(vnd_range.group(2).replace(",", "."))
        return (a, b, (a + b) / 2)

    vnd_single = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:triệu|tr)\b", combined)
    if vnd_single:
        val = float(vnd_single.group(1).replace(",", "."))
        return (val, val, val)

    return (None, None, None)


def extract_experience(title: str, text: str) -> float:
    title = "" if pd.isna(title) else str(title)
    text = "" if pd.isna(text) else str(text)
    combined = (title + " " + extract_real_content(text))[:6000].lower()

    fresher_patterns = [
        r"\bfresher\b",
        r"fresh\s+graduate",
        r"không\s+yêu\s+cầu\s+kinh\s+nghiệm",
        r"chưa\s+có\s+kinh\s+nghiệm",
        r"không\s+cần\s+kinh\s+nghiệm",
    ]
    if any(re.search(p, combined) for p in fresher_patterns):
        return 0.0

    range_match = re.search(r"(\d+)\s*[-~đến]+\s*(\d+)\s*(?:năm|years?)", combined)
    if range_match:
        a = float(range_match.group(1))
        b = float(range_match.group(2))
        return (a + b) / 2

    plus_match = re.search(r"(\d+)\+\s*(?:năm|years?)", combined)
    if plus_match:
        val = float(plus_match.group(1))
        if 0 <= val <= 20:
            return val

    single_match = re.search(r"(\d+)\s*(?:năm|years?)\s*(?:kinh\s*nghiệm|experience)?", combined)
    if single_match:
        val = float(single_match.group(1))
        if 0 <= val <= 20:
            return val

    return None


def extract_skills(title: str, description: str):
    title_text = "" if pd.isna(title) else str(title).lower()
    desc_cleaned = extract_real_content(description)
    combined = f"{title_text} {desc_cleaned}"

    found = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, combined, re.IGNORECASE):
            found.append(skill)
    return found


def main():
    print(f"\nDang doc du lieu tu: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    print(f"Da doc xong: {df.shape[0]} dong, {df.shape[1]} cot")

    print("\nPhan tich mau noi dung:")
    sample_orig = str(df.iloc[0]["noi_dung_clean"])
    sample_clean = extract_real_content(sample_orig)
    print(f"Do dai ban goc: {len(sample_orig):,} ky tu")
    print(f"Do dai sau khi loc: {len(sample_clean):,} ky tu")
    print(f"Vi du: {sample_clean[:150]}...")

    lens = df["noi_dung_clean"].apply(lambda x: len(extract_real_content(x)))
    print("\nThong ke do dai noi dung sau khi loc:")
    print(lens.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    print("So dong qua ngan (<800):", int((lens < 800).sum()), "/", int(len(lens)))

    print("\nDang nhom cong viec (job_group)...")
    df["job_group"] = df["tieu_de_clean"].apply(detect_job_group)
    group_counts = df["job_group"].value_counts()
    other_count = int(group_counts.get("other", 0))
    other_pct = other_count / len(df) * 100
    print(f"So nhom: {len(group_counts)}")
    print(f"Nhom other: {other_count}/{len(df)} ({other_pct:.1f}%)")

    print("\nDang trich xuat thong tin luong...")
    salary = df.apply(lambda r: extract_salary(r["tieu_de_clean"], r["noi_dung_clean"]), axis=1)
    df["salary_min"] = salary.apply(lambda x: x[0])
    df["salary_max"] = salary.apply(lambda x: x[1])
    df["salary_avg"] = salary.apply(lambda x: x[2])
    df["has_salary"] = df["salary_avg"].notna().astype(int)
    n_salary = int(df["has_salary"].sum())
    print(f"So tin co luong: {n_salary}/{len(df)} ({n_salary / len(df) * 100:.1f}%)")

    print("\nDang trich xuat kinh nghiem...")
    df["experience_years"] = df.apply(lambda r: extract_experience(r["tieu_de_clean"], r["noi_dung_clean"]), axis=1)
    n_exp = int(df["experience_years"].notna().sum())
    if n_exp > 0:
        median_exp = df["experience_years"].median()
        mean_exp = df["experience_years"].mean()
        print(f"So tin co kinh nghiem: {n_exp}/{len(df)} ({n_exp / len(df) * 100:.1f}%)")
        print(f"Trung vi: {median_exp:.1f} nam, Trung binh: {mean_exp:.1f} nam")
    else:
        print("Khong tim thay thong tin kinh nghiem")

    print("\nDang trich xuat ky nang (tu title + noi dung da loc)...")
    df["skills"] = df.apply(lambda r: extract_skills(r["tieu_de_clean"], r["noi_dung_clean"]), axis=1)
    df["n_skills"] = df["skills"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda lst: ",".join(lst))

    avg_skills = float(df["n_skills"].mean())
    print(f"So ky nang trung binh moi tin: {avg_skills:.2f}")

    all_skills = []
    for skills in df["skills"]:
        all_skills.extend(skills)
    skill_counts = Counter(all_skills)

    print("\nTop 20 ky nang:")
    for skill, count in skill_counts.most_common(20):
        pct = count / len(df) * 100
        if pct > 90:
            status = "QUA CAO"
        elif pct > 50:
            status = "CAO"
        else:
            status = "BINH THUONG"
        print(f"{status:12s} {skill:15s}: {count:4d}/{len(df)} ({pct:5.1f}%)")

    print(f"\nDang luu ket qua ra: {FEATURES_PATH}")
    df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"Da luu xong: {df.shape[0]} dong, {df.shape[1]} cot")

    print("\nTom tat:")
    print(f"Tong so tin: {len(df)}")
    print(f"So nhom job_group: {len(group_counts)}")
    print(f"Ty le other: {other_pct:.1f}%")
    print(f"So tin co luong: {n_salary} ({n_salary / len(df) * 100:.1f}%)")
    print(f"So tin co kinh nghiem: {n_exp} ({n_exp / len(df) * 100:.1f}%)")
    print(f"So ky nang trung binh: {avg_skills:.2f}")
    print(f"So ky nang khac nhau: {len(skill_counts)}")

    print("\nKiem tra nhanh:")
    top_pct = (skill_counts.most_common(1)[0][1] / len(df) * 100) if skill_counts else 0

    if avg_skills >= 2.5:
        print(f"So ky nang trung binh dat yeu cau: {avg_skills:.2f}")
    elif avg_skills >= 1.5:
        print(f"So ky nang trung binh tam chap nhan: {avg_skills:.2f}")
    else:
        print(f"So ky nang trung binh thap: {avg_skills:.2f}")

    if top_pct > 90:
        print(f"Ky nang top bi qua cao: {top_pct:.1f}%")
    elif top_pct > 50:
        print(f"Ky nang top kha cao: {top_pct:.1f}%")
    else:
        print(f"Ky nang top binh thuong: {top_pct:.1f}%")

    if n_exp > len(df) * 0.3:
        print(f"Ty le co kinh nghiem dat yeu cau: {n_exp / len(df) * 100:.1f}%")
    else:
        print(f"Ty le co kinh nghiem thap: {n_exp / len(df) * 100:.1f}%")


    print("\nVi du 3 dong dau:")
    cols = ["tieu_de", "job_group", "n_skills", "skills_str", "experience_years"]
    available = [c for c in cols if c in df.columns]
    print(df[available].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
