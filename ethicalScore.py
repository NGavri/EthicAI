def calculate_ethical_score(fairness_metrics, privacy_metrics):
    score = {}

    # 1. Fairness Score
    spd = abs(fairness_metrics.get("Statistical Parity Difference", 1))
    eod = abs(fairness_metrics.get("Equal Opportunity Difference", 1))
    di = fairness_metrics.get("Disparate Impact", 0)
    group_acc = fairness_metrics.get("Group-wise Accuracy", {})

    # Normalize
    spd_score = max(0, 1 - spd * 10)
    eod_score = max(0, 1 - eod * 10)
    di_score = 1 if 0.8 <= di <= 1.25 else max(0, 1 - abs(1 - di))
    if isinstance(group_acc, dict) and len(group_acc) > 1:
        acc_gap = max(group_acc.values()) - min(group_acc.values())
        acc_score = max(0, 1 - acc_gap * 5)
    else:
        acc_score = 0.5

    fairness_score = 0.5 * (spd_score + eod_score + di_score + acc_score) / 4

    # 2. Privacy Score
    mia_gap = privacy_metrics["Membership Inference Risk"]["Details"].get("Confidence Gap", 1)
    mia_score = max(0, 1 - mia_gap * 4)

    leakage_percent = float(
        privacy_metrics["Data Leakage Check"]["Details"].get("Leakage Percentage", "100%").replace('%', '')
    )
    leakage_score = max(0, 1 - leakage_percent / 20)

    dp_score = 1 if "Privacy Preserved" in privacy_metrics["Differential Privacy Check"]["Risk Level"] else 0.5

    privacy_score = 0.5 * (mia_score + leakage_score + dp_score) / 3

    # Total Score
    total_score = round(fairness_score + privacy_score, 2)

    # Label
    if total_score >= 0.85:
        label = "Highly Ethical"
    elif total_score >= 0.7:
        label = "Moderately Ethical"
    else:
        label = "Needs Improvement"

    return {
        "Ethical Score": total_score,
        "Interpretation": label,
        "Breakdown": {
            "Fairness": round(fairness_score, 2),
            "Privacy": round(privacy_score, 2)
        }
    }