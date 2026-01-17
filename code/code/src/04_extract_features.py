# src/04_extract_features.py
import re
import pandas as pd
from config import CLEAN_PATH, FEATURES_PATH

# -------------- 1) TỪ ĐIỂN KỸ NĂNG (bạn có thể mở rộng) --------------
SKILL_PATTERNS = {
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "c++": r"\bc\+\+\b",
    "c#": r"\bc\#\b",
    "javascript": r"\bjavascript\b|\bjs\b",
    "typescript": r"\btypescript\b|\bts\b",
    "react": r"\breact\b|\breactjs\b",
    "nodejs": r"\bnode\.?js\b|\bnodejs\b",
    "php": r"\bphp\b",
    ".net": r"\b\.net\b|\bdotnet\b",
    "sql": r"\bsql\b",
    "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgres\b|\bpostgresql\b",
    "mongodb": r"\bmongodb\b|\bmongo\b",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    "git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "linux": r"\blinux\b|\bubuntu\b|\bcentos\b",
    "html": r"\bhtml\b",
    "css": r"\bcss\b",
    "excel": r"\bexcel\b",
    "powerbi": r"\bpower\s?bi\b",
    "tableau": r"\btableau\b",
}

# -------------- 2) NHÓM VỊ TRÍ (từ tieu_de) --------------
JOB_GROUP_RULES = [
    ("data", r"\bdata\b|\banalyst\b|\bai\b|\bml\b|\bmachine learning\b|\bds\b"),
    ("backend", r"\bbackend\b|\bback-end\b|\bapi\b"),
    ("frontend", r"\bfrontend\b|\bfront-end\b|\breact\b|\bvue\b|\bangular\b"),
    ("fullstack", r"\bfullstack\b|\bfull-stack\b"),
    ("devops", r"\bdevops\b|\bsre\b|\bcloud\b|\baws\b|\bkubernetes\b"),
    ("qa", r"\bqa\b|\btester\b|\btest\b"),
    ("mobile", r"\bandroid\b|\bios\b|\bflutter\b|\breact native\b"),
]

def detect_job_group(title_clean: str) -> str:
    for group, pattern in JOB_GROUP_RULES:
        if re.search(pattern, title_clean):
            return group
    return "other"

# -------------- 3) TRÍCH XUẤT LƯƠNG (regex cơ bản) --------------
def extract_salary(text: str):
    """
    Trả về (min_million, max_million, avg_million)
    Nếu không thấy -> (None, None, None)
    """
    t = text.lower()

    # nếu có từ khóa thỏa thuận
    if "thỏa thuận" in t or "thoả thuận" in t:
        return (None, None, None)

    # vd: 15 - 25 triệu
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*[-~đến]+\s*(\d+(?:[\.,]\d+)?)\s*(triệu|tr)", t)
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        return (a, b, (a + b) / 2)

    # vd: lương 20 triệu
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(triệu|tr)\b", t)
    if m:
        a = float(m.group(1).replace(",", "."))
        return (a, a, a)

    return (None, None, None)

# -------------- 4) TRÍCH XUẤT KỸ NĂNG --------------
def extract_skills(text: str):
    found = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text):
            found.append(skill)
    return found

def main():
    df = pd.read_csv(CLEAN_PATH)

    df["job_group"] = df["tieu_de_clean"].apply(detect_job_group)

    # salary
    salary = df["noi_dung_clean"].apply(extract_salary)
    df["salary_min"] = salary.apply(lambda x: x[0])
    df["salary_max"] = salary.apply(lambda x: x[1])
    df["salary_avg"] = salary.apply(lambda x: x[2])
    df["has_salary"] = df["salary_avg"].notna().astype(int)

    # skills
    df["skills"] = df["noi_dung_clean"].apply(extract_skills)
    df["n_skills"] = df["skills"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda lst: ",".join(lst))

    df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print("✅ Saved:", FEATURES_PATH, "| Shape:", df.shape)
    print(df[["tieu_de", "job_group", "skills_str", "salary_avg"]].head(5))

if __name__ == "__main__":
    main()
