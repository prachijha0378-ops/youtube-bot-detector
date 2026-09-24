"""
detector.py — Isolation Forest YouTube Bot Detector
CDAC Final Project — v3 (context-aware features)

Key insight:
  BOT pattern  = HIGH subs + LOW views  (bought fake subscribers)
  NORMAL small = LOW subs  + HIGH views (viral video, organic)
  NORMAL large = HIGH subs + HIGH views (established creator)
"""

import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURES — context-aware, bot-specific signals
# ============================================================
FEATURE_KEYS = [
    "sub_engagement_deficit",   # subs >> views  (main bot signal)
    "views_per_video_score",    # views per video vs subscriber size
    "upload_velocity",          # videos per day
    "sub_growth_suspicion",     # many subs, very new channel
    "log_subscribers",          # channel scale
    "log_view_sub_ratio",       # log(views/subs) — natural range
    "consistency_score",        # subs, views, age all aligned?
]

FEAT_DISPLAY_STATS = {
    "sub_engagement_deficit":  {"mean": 0.0,  "std": 1.0},
    "views_per_video_score":   {"mean": 0.0,  "std": 1.0},
    "upload_velocity":         {"mean": 0.15, "std": 0.25},
    "sub_growth_suspicion":    {"mean": 0.0,  "std": 1.0},
    "log_subscribers":         {"mean": 10.5, "std": 2.5},
    "log_view_sub_ratio":      {"mean": 3.5,  "std": 1.5},
    "consistency_score":       {"mean": 0.0,  "std": 1.0},
}


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================
def _train():
    np.random.seed(42)
    n = 6000

    # ── Normal channels (organic) ──
    # Small organic: low subs, decent views (viral possible)
    n1 = n // 3
    subs1  = np.random.lognormal(6, 1.5, n1).clip(50, 10000)
    views1 = subs1 * np.random.lognormal(2, 1.5, n1).clip(0.2, 500)
    vids1  = np.random.lognormal(3, 1.2, n1).clip(1, 500)
    age1   = np.random.lognormal(5.5, 0.8, n1).clip(30, 2000)

    # Medium organic
    n2 = n // 3
    subs2  = np.random.lognormal(11, 1.5, n2).clip(10000, 2000000)
    views2 = subs2 * np.random.lognormal(3, 0.8, n2).clip(5, 300)
    vids2  = np.random.lognormal(5.5, 1, n2).clip(20, 3000)
    age2   = np.random.lognormal(7, 0.7, n2).clip(300, 4000)

    # Large organic
    n3 = n - n1 - n2
    subs3  = np.random.lognormal(14, 1.5, n3).clip(1000000, 100000000)
    views3 = subs3 * np.random.lognormal(4, 0.7, n3).clip(20, 500)
    vids3  = np.random.lognormal(7, 0.8, n3).clip(100, 10000)
    age3   = np.random.lognormal(7.8, 0.6, n3).clip(800, 6000)

    subs  = np.concatenate([subs1, subs2, subs3])
    views = np.concatenate([views1, views2, views3])
    vids  = np.concatenate([vids1, vids2, vids3])
    age   = np.concatenate([age1, age2, age3])

    # Feature engineering
    vpv    = views / np.maximum(vids, 1)
    vpday  = vids  / np.maximum(age, 1)
    log_s  = np.log1p(subs)
    log_vsr= np.log1p(views / np.maximum(subs, 1))  # log(views/subs)

    # sub_engagement_deficit: high when subs >> views (BOT signal)
    # = log(subs) - log(views+1)  positive means more subs than views
    sed = log_s - np.log1p(views)

    # views_per_video_score: compare vpv relative to subscriber size
    # large channels should have high vpv; small can have low
    # score = log(vpv) - 0.5*log(subs)  (size-adjusted)
    vpvs = np.log1p(vpv) - 0.5 * log_s

    # sub_growth_suspicion: subs/age — too many subs for age
    sgs = np.log1p(subs / np.maximum(age, 1))

    # consistency: subs, views, age naturally correlated
    # log(views) should ≈ log(subs) + some constant
    cons = np.log1p(views) - log_s

    X = np.column_stack([sed, vpvs, vpday, sgs, log_s, log_vsr, cons])

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(Xs)
    return iso, sc


_MODEL, _SCALER = _train()


# ============================================================
# FEATURE EXTRACTION FROM YOUTUBE API DATA
# ============================================================
def _extract(channel, videos):
    stats   = channel.get("statistics", {})
    snippet = channel.get("snippet", {})

    try:    subs  = max(int(stats.get("subscriberCount", 0)), 0)
    except: subs  = 0
    try:    views = max(int(stats.get("viewCount", 0)), 0)
    except: views = 0
    try:    vids  = max(int(stats.get("videoCount", 1)), 1)
    except: vids  = 1

    pub = snippet.get("publishedAt", "")
    try:
        created = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        age = max(1, (datetime.now(created.tzinfo) - created).days)
    except:
        age = 365

    # Recent upload velocity from actual video timestamps
    vpday = vids / max(age, 1)
    if videos and len(videos) >= 2:
        dates = []
        for v in videos:
            p = v.get("snippet", {}).get("publishedAt")
            if p:
                try:
                    dates.append(
                        datetime.fromisoformat(p.replace("Z", "+00:00"))
                    )
                except:
                    pass
        if len(dates) >= 2:
            dates.sort()
            span = max(1, (dates[-1] - dates[0]).total_seconds() / 86400)
            recent_vpd = len(dates) / span
            vpday = (vpday + recent_vpd) / 2

    log_s   = np.log1p(subs)
    vpv     = views / max(vids, 1)
    log_vsr = np.log1p(views / max(subs, 1))

    # Context-aware features
    sed  = log_s - np.log1p(views)                    # engagement deficit
    vpvs = np.log1p(vpv) - 0.5 * log_s               # size-adjusted vpv
    sgs  = np.log1p(subs / max(age, 1))               # sub growth rate
    cons = np.log1p(views) - log_s                     # view-sub consistency

    return {
        "sub_engagement_deficit":  sed,
        "views_per_video_score":   vpvs,
        "upload_velocity":         vpday,
        "sub_growth_suspicion":    sgs,
        "log_subscribers":         log_s,
        "log_view_sub_ratio":      log_vsr,
        "consistency_score":       cons,
        # raw values
        "_subs": subs, "_views": views, "_vids": vids,
        "_age": age, "_vpv": vpv, "_vpday": vpday,
    }


# ============================================================
# HUMAN-READABLE SIGNALS
# ============================================================
def _signals(f, risk):
    sigs  = []
    subs  = f["_subs"]
    views = f["_views"]
    vids  = f["_vids"]
    age   = f["_age"]
    vpv   = f["_vpv"]
    vpday = f["_vpday"]

    # PRIMARY BOT SIGNAL: bought subscribers
    if subs > 0 and views < subs * 0.3:
        pct = (views / subs) * 100
        sigs.append(
            f"🚨 Total views ({views:,}) only {pct:.1f}% of subscribers ({subs:,}). "
            f"Subscribers likely purchased — real audiences watch videos."
        )
    elif subs > 0 and views < subs * 0.8:
        sigs.append(
            f"⚠️ Views ({views:,}) significantly lower than subscribers ({subs:,}). "
            f"Possible subscriber inflation."
        )

    # LOW ENGAGEMENT PER VIDEO despite large audience
    if subs > 50000 and vpv < subs * 0.005:
        sigs.append(
            f"🚨 Only {vpv:.0f} avg views/video for {subs:,} subscribers "
            f"({(vpv/subs*100):.2f}% engagement). Very suspicious."
        )
    elif subs > 10000 and vpv < subs * 0.01:
        sigs.append(
            f"⚠️ Low avg views/video ({vpv:.0f}) relative to {subs:,} subscribers."
        )

    # UPLOAD AUTOMATION
    if vpday > 15:
        sigs.append(
            f"🚨 {vpday:.1f} videos/day — almost certainly automated bot content."
        )
    elif vpday > 5:
        sigs.append(
            f"⚠️ {vpday:.1f} videos/day — unusually high upload rate."
        )

    # FAST SUBSCRIBER GROWTH on new channel with low views
    if age < 200 and subs > 100000 and views < subs:
        sigs.append(
            f"🚨 {subs:,} subscribers in only {age} days but views < subs — "
            f"rapid fake subscriber purchase suspected."
        )
    elif age < 365 and subs > 500000 and views > subs * 5:
        sigs.append(
            f"⚠️ Very fast organic growth — {subs:,} subs in {age} days. "
            f"Unusual but views support it."
        )

    # ALL GOOD
    if not sigs:
        if risk < 30:
            sigs.append(
                "✅ No bot signals detected. Channel metrics are consistent "
                "with organic YouTube growth."
            )
        else:
            sigs.append(
                "⚠️ Some unusual patterns but no definitive bot signals. "
                "Could be a niche or new channel. Monitor over time."
            )

    return sigs


# ============================================================
# MAIN ANALYZE FUNCTION
# ============================================================
def analyze_channel(channel, videos):
    f  = _extract(channel, videos)
    x  = np.array([[f[k] for k in FEATURE_KEYS]])
    xs = _SCALER.transform(x)

    pred = _MODEL.predict(xs)[0]
    raw  = float(_MODEL.decision_function(xs)[0])

    # Convert to 0-100 risk
    risk = max(0, min(100, int(((-raw + 0.04) / 0.22) * 100)))

    subs  = f["_subs"]
    views = f["_views"]
    vpday = f["_vpday"]

    # ── Step 1: Reduce risk for clearly normal channels ──
    # Only apply reductions when upload rate is sane (< 3/day)

    if vpday < 2:
        # Very small channel (< 500 subs) — almost never a bot
        if subs < 500:
            risk = 0

        # Small channel with viral content = organic, NOT bot
        if subs < 5000 and views > subs * 5:
            risk = 0

        # Old established channel with strong engagement
        if f["_age"] > 1000 and views > subs * 10:
            risk = 0

        # New small channel with normal pattern
        if subs < 2000 and views >= subs:
            risk = min(risk, 10)

        # Any channel where views >> subs and uploads normal
        if views > subs * 3:
            risk = min(risk, 10)

        # Good views per video
        vpv_local = f["_vpv"]
        if vpv_local > 1000:
            risk = min(risk, 10)

    # ── MEGA CHANNEL protection ──
    # Large established channels (T-Series, MrBeast etc.)
    # High upload rate is NORMAL for big media companies
    if subs > 1000000 and views > subs * 50 and f["_age"] > 1000:
        risk = min(risk, 20)   # very established, high engagement
    if subs > 10000000 and views > subs * 100:
        risk = min(risk, 15)   # mega channel, massive engagement
    if subs > 50000000:
        risk = min(risk, 25)   # extremely large channel

    # ── Step 2: OVERRIDE for clear bot signals (applied LAST, cannot be reduced) ──

    # HARD BOT: subscribers >> views (fake subs bought)
    # Only flag if views are LOW relative to subs — not if both are high
    if subs > 10000 and views < subs * 0.3:
        risk = max(risk, 80)
    elif subs > 5000 and views < subs * 0.5:
        risk = max(risk, 65)

    # HARD BOT: extreme upload rate — but NOT for large established channels
    # Big media companies (news, music) upload many videos daily legitimately
    if vpday > 15 and subs < 1000000:
        risk = max(risk, 82)
    elif vpday > 5 and subs < 500000:
        risk = max(risk, 70)
    elif vpday > 3 and subs < 100000:
        risk = max(risk, 68)

    risk = max(0, min(100, risk))

    if risk >= 65:   verdict = "HIGH ANOMALY RISK"
    elif risk >= 45: verdict = "SUSPICIOUS"
    else:            verdict = "LIKELY NORMAL"

    sigs = _signals(f, risk)

    vps_display = round(views / max(subs, 1), 2)
    eng_rate    = f"{(f['_vpv']/max(subs,1)*100):.2f}%" if subs > 0 else "N/A"

    display = {
        "Channel Age (days)":   round(f["_age"]),
        "Subscribers":          f["_subs"],
        "Total Views":          f["_views"],
        "Total Videos":         f["_vids"],
        "Views / Subscriber":   vps_display,
        "Avg Views / Video":    round(f["_vpv"]),
        "Videos / Day":         round(f["_vpday"], 3),
        "Engagement Rate":      eng_rate,
    }

    # Z-scores using same display keys
    z_display = {
        "Channel Age (days)":   round((f["_age"] - 1800) / 900, 2),
        "Subscribers":          round((np.log1p(subs) - 10.5) / 2.5, 2),
        "Total Views":          round((np.log1p(views) - 14.0) / 2.5, 2),
        "Total Videos":         round((np.log1p(f["_vids"]) - 5.0) / 1.5, 2),
        "Views / Subscriber":   round((vps_display - 50) / 80, 2),
        "Avg Views / Video":    round((f["_vpv"] - 40000) / 80000, 2),
        "Videos / Day":         round((f["_vpday"] - 0.15) / 0.25, 2),
        "Engagement Rate":      0.0,
    }

    return {
        "score":        risk,
        "verdict":      verdict,
        "signals":      sigs,
        "features":     display,
        "z_scores":     z_display,
        "ml_label":     int(pred),
        "raw_if_score": round(raw, 4),
    }
