"""
src/send_alerts.py  —  Email Alert System
------------------------------------------
Sends ONE summary email via Gmail SMTP whenever Wildfire Risk hotspots
are detected after model classification.

Usage (standalone):
    python src/send_alerts.py                  # alerts on features_labeled.csv
    python src/send_alerts.py --frp 50        # only hotspots with FRP >= 50 MW
    python src/send_alerts.py --dry-run       # preview without sending

Called automatically by train_model.py after Step 4.

Environment variables required (set in .env):
    ALERT_EMAIL_ENABLED    = true
    SMTP_USER              = you@gmail.com
    SMTP_PASSWORD          = xxxx xxxx xxxx xxxx   ← Gmail App Password
    ALERT_RECIPIENTS       = dest@example.com,another@example.com
    ALERT_FRP_THRESHOLD    = 0    (0 = send all Wildfire Risk, or set e.g. 50)
"""

import argparse
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "features_labeled.csv"
SENT_LOG  = ROOT / "data" / "alerts_sent.log"   # deduplication log

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Config — read from environment
# ---------------------------------------------------------------------------
SMTP_HOST          = os.getenv("SMTP_HOST",           "smtp.gmail.com")
SMTP_PORT          = int(os.getenv("SMTP_PORT",       "587"))
SMTP_USER          = os.getenv("SMTP_USER",           "")
SMTP_PASSWORD      = os.getenv("SMTP_PASSWORD",       "").replace(" ", "")  # strip spaces — Gmail App Passwords are 16 chars
ALERT_ENABLED      = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
RECIPIENTS_RAW     = os.getenv("ALERT_RECIPIENTS",    "")
FRP_THRESHOLD      = float(os.getenv("ALERT_FRP_THRESHOLD", "0"))

RECIPIENTS: list[str] = [
    r.strip() for r in RECIPIENTS_RAW.split(",") if r.strip()
]


# ---------------------------------------------------------------------------
# Google Maps helper
# ---------------------------------------------------------------------------
def maps_link(lat: float, lon: float) -> str:
    """Return a Google Maps URL for the given coordinates."""
    return f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"


# ---------------------------------------------------------------------------
# HTML email builder
# ---------------------------------------------------------------------------
def build_html_email(hotspots: pd.DataFrame, threshold: float) -> str:
    """
    Build a rich HTML email body listing all Wildfire Risk hotspots.
    Returns the complete HTML string.
    """
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    count    = len(hotspots)
    thresh_note = (
        f" (FRP ≥ {threshold:.0f} MW filter applied)" if threshold > 0 else ""
    )

    # ── Table rows ──────────────────────────────────────────────────────────
    rows_html = ""
    for rank, (_, row) in enumerate(hotspots.iterrows(), start=1):
        lat   = row.get("latitude",  float("nan"))
        lon   = row.get("longitude", float("nan"))
        frp   = row.get("frp",       float("nan"))
        date  = str(row.get("acq_date", ""))[:10]
        time_ = str(row.get("acq_time", ""))[:4]
        lu    = str(row.get("land_use_type", "unknown")).capitalize()
        freq  = row.get("historical_frequency", 0)
        tod   = str(row.get("time_of_day", ""))
        season= str(row.get("season", ""))

        try:
            frp_str = f"{float(frp):.1f} MW"
        except (TypeError, ValueError):
            frp_str = "N/A"

        try:
            coord_str = f"{float(lat):.4f}°N, {float(lon):.4f}°E"
            link      = maps_link(float(lat), float(lon))
            map_btn   = (
                f'<a href="{link}" style="'
                'background:#dc2626;color:#fff;padding:3px 10px;'
                'border-radius:4px;text-decoration:none;font-size:12px;'
                'font-weight:600;">📍 View Map</a>'
            )
        except (TypeError, ValueError):
            coord_str = f"{lat}, {lon}"
            map_btn   = "—"

        # Alternate row shade
        bg = "#fff7f7" if rank % 2 == 0 else "#ffffff"

        rows_html += f"""
        <tr style="background:{bg};">
          <td style="padding:10px 14px;font-weight:700;color:#dc2626;
                     font-size:15px;text-align:center;">#{rank}</td>
          <td style="padding:10px 14px;font-weight:700;color:#b91c1c;
                     font-size:14px;">{frp_str}</td>
          <td style="padding:10px 14px;font-size:13px;color:#374151;">
            {coord_str}
          </td>
          <td style="padding:10px 14px;font-size:13px;color:#374151;">
            {date}&nbsp;{time_}
          </td>
          <td style="padding:10px 14px;font-size:13px;color:#374151;">{lu}</td>
          <td style="padding:10px 14px;font-size:13px;color:#374151;">
            {tod} · {season}
          </td>
          <td style="padding:10px 14px;font-size:13px;color:#374151;">
            {int(float(freq)) if str(freq).replace('.','').isdigit() else freq}
          </td>
          <td style="padding:10px 14px;text-align:center;">{map_btn}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>🔥 Wildfire Risk Alert</title>
</head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Inter,
             Arial,sans-serif;">

  <!-- Header banner -->
  <div style="background:linear-gradient(135deg,#dc2626,#991b1b);
              padding:28px 32px;text-align:center;">
    <div style="font-size:40px;margin-bottom:8px;">🔥</div>
    <h1 style="margin:0;color:#ffffff;font-size:24px;font-weight:700;
               letter-spacing:0.5px;">
      Wildfire Risk Alert — India
    </h1>
    <p style="margin:8px 0 0;color:#fca5a5;font-size:14px;">
      AI-Based Industrial Fire Detection System · Generated {now_str}
    </p>
  </div>

  <!-- Summary card -->
  <div style="max-width:860px;margin:24px auto;background:#fff;
              border-radius:10px;border:1px solid #e5e7eb;
              box-shadow:0 2px 8px rgba(0,0,0,.06);overflow:hidden;">

    <div style="padding:20px 28px;border-bottom:1px solid #fee2e2;
                background:#fff7f7;">
      <p style="margin:0;font-size:16px;color:#111827;">
        <strong style="color:#dc2626;">{count} Wildfire Risk hotspot(s)</strong>
        detected in the latest satellite pass{thresh_note}.
      </p>
      <p style="margin:6px 0 0;font-size:13px;color:#6b7280;">
        Source: NASA FIRMS (VIIRS / MODIS) · Classified by XGBoost model
      </p>
    </div>

    <!-- Data table -->
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#fef2f2;">
            <th style="padding:10px 14px;text-align:center;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">#</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">FRP Intensity</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">Coordinates</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">Date / Time</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">Land Use</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">Time · Season</th>
            <th style="padding:10px 14px;text-align:left;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">Repeat Count</th>
            <th style="padding:10px 14px;text-align:center;color:#6b7280;
                       font-size:12px;font-weight:600;border-bottom:
                       2px solid #fee2e2;">Map</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <!-- Footer note -->
    <div style="padding:16px 28px;background:#f9fafb;border-top:
                1px solid #e5e7eb;">
      <p style="margin:0;font-size:12px;color:#9ca3af;line-height:1.6;">
        <strong>FRP</strong> = Fire Radiative Power (MW) — higher = more intense fire.
        <strong>Repeat Count</strong> = times this ~1 km location was flagged as a hotspot.
        Coordinates link to Google Maps for field verification.
        <br>This alert was generated automatically by the SIH AI Fire Detection pipeline.
      </p>
    </div>
  </div>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Plain-text fallback (for email clients that don't render HTML)
# ---------------------------------------------------------------------------
def build_text_email(hotspots: pd.DataFrame, threshold: float) -> str:
    count      = len(hotspots)
    now_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    thresh_note = f" (FRP >= {threshold:.0f} MW)" if threshold > 0 else ""

    lines = [
        "=" * 60,
        "  WILDFIRE RISK ALERT — AI Fire Detection System",
        "=" * 60,
        f"  Generated : {now_str}",
        f"  Hotspots  : {count} Wildfire Risk detection(s){thresh_note}",
        "=" * 60,
        "",
    ]

    for rank, (_, row) in enumerate(hotspots.iterrows(), start=1):
        lat   = row.get("latitude",  "?")
        lon   = row.get("longitude", "?")
        frp   = row.get("frp",       "?")
        date  = str(row.get("acq_date", ""))[:10]
        lu    = str(row.get("land_use_type", "unknown"))

        try:
            frp_str = f"{float(frp):.1f} MW"
        except (TypeError, ValueError):
            frp_str = str(frp)

        try:
            link = maps_link(float(lat), float(lon))
            coord = f"{float(lat):.4f}N, {float(lon):.4f}E"
        except (TypeError, ValueError):
            link  = "N/A"
            coord = f"{lat}, {lon}"

        lines += [
            f"  #{rank}  FRP: {frp_str}  |  {coord}  |  {date}  |  {lu}",
            f"       Google Maps: {link}",
            "",
        ]

    lines += [
        "-" * 60,
        "  FRP = Fire Radiative Power (MW).",
        "  Generated by SIH AI Fire Detection pipeline.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core send function
# ---------------------------------------------------------------------------
def send_alert_email(
    hotspots: pd.DataFrame,
    threshold: float = 0.0,
    dry_run: bool = False,
) -> bool:
    """
    Send ONE summary email listing all Wildfire Risk hotspots.

    Parameters
    ----------
    hotspots  : DataFrame — already filtered to Wildfire Risk rows, sorted by FRP desc.
    threshold : float     — FRP threshold used (for display in email body only).
    dry_run   : bool      — if True, print the email but do NOT send it.

    Returns
    -------
    bool — True if email was sent (or dry-run succeeded), False on error.
    """
    if hotspots.empty:
        print("  [Alerts] No Wildfire Risk hotspots — nothing to send.")
        return True

    if not ALERT_ENABLED and not dry_run:
        print("  [Alerts] ALERT_EMAIL_ENABLED is not 'true' in .env — skipping.")
        return True

    if not SMTP_USER or not SMTP_PASSWORD:
        print("  [Alerts] SMTP_USER / SMTP_PASSWORD not set in .env — skipping.")
        return False

    if not RECIPIENTS:
        print("  [Alerts] ALERT_RECIPIENTS not set in .env — skipping.")
        return False

    count   = len(hotspots)
    subject = (
        f"🔥 [{count} ALERT{'S' if count > 1 else ''}] "
        f"Wildfire Risk Detected — AI Fire Detection"
    )

    # Build message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"AI Fire Detection <{SMTP_USER}>"
    msg["To"]      = ", ".join(RECIPIENTS)

    text_part = MIMEText(build_text_email(hotspots, threshold), "plain", "utf-8")
    html_part = MIMEText(build_html_email(hotspots, threshold), "html",  "utf-8")

    msg.attach(text_part)   # plain first — HTML takes priority in modern clients
    msg.attach(html_part)

    if dry_run:
        print()
        print("  ── DRY RUN — email NOT sent ──────────────────────────────")
        print(f"  To      : {', '.join(RECIPIENTS)}")
        print(f"  Subject : {subject}")
        print(f"  Hotspots: {count}")
        print()
        print(build_text_email(hotspots, threshold))
        return True

    # ── Send via Gmail SMTP (TLS) ────────────────────────────────────────
    try:
        print(f"  [Alerts] Connecting to {SMTP_HOST}:{SMTP_PORT} …")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, RECIPIENTS, msg.as_string())
        print(f"  [Alerts] ✅ Email sent to: {', '.join(RECIPIENTS)}")
        _write_sent_log(count, threshold)
        return True

    except smtplib.SMTPAuthenticationError:
        print(
            "  [Alerts] ❌ SMTP Authentication failed.\n"
            "           Make sure you are using a Gmail App Password,\n"
            "           not your regular Gmail password.\n"
            "           Guide: https://myaccount.google.com/apppasswords"
        )
        return False

    except smtplib.SMTPException as exc:
        print(f"  [Alerts] ❌ SMTP error: {exc}")
        return False

    except OSError as exc:
        print(f"  [Alerts] ❌ Network error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Deduplication log
# ---------------------------------------------------------------------------
def _write_sent_log(count: int, threshold: float) -> None:
    """Append a line to data/alerts_sent.log after a successful send."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    SENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"{now_str}  |  hotspots={count}  |  frp_threshold={threshold}\n")


# ---------------------------------------------------------------------------
# Main entry point — called by train_model.py or standalone
# ---------------------------------------------------------------------------
def send_alerts(
    df: pd.DataFrame | None = None,
    frp_threshold: float | None = None,
    dry_run: bool = False,
) -> bool:
    """
    Check `df` (or load features_labeled.csv) for Wildfire Risk rows
    and send ONE summary alert email.

    Parameters
    ----------
    df            : DataFrame to scan. If None, loads features_labeled.csv.
    frp_threshold : Override ALERT_FRP_THRESHOLD from .env.
    dry_run       : Preview without sending.

    Returns
    -------
    bool — True on success / skipped, False on error.
    """
    # ── Load data ────────────────────────────────────────────────────────
    if df is None:
        if not DATA_FILE.exists():
            print(f"  [Alerts] Data file not found: {DATA_FILE}")
            return False
        df = pd.read_csv(DATA_FILE)

    # ── Resolve FRP threshold ────────────────────────────────────────────
    threshold = frp_threshold if frp_threshold is not None else FRP_THRESHOLD

    # ── Determine category column ────────────────────────────────────────
    cat_col = "predicted_category" if "predicted_category" in df.columns else "category"

    # ── Filter to Wildfire Risk ──────────────────────────────────────────
    wildfire_df = df[df[cat_col] == "Wildfire Risk"].copy()

    # ── Apply FRP threshold if set ───────────────────────────────────────
    if threshold > 0 and "frp" in wildfire_df.columns:
        wildfire_df = wildfire_df[wildfire_df["frp"] >= threshold]

    # ── Sort by FRP descending ───────────────────────────────────────────
    if "frp" in wildfire_df.columns:
        wildfire_df = wildfire_df.sort_values("frp", ascending=False)

    # ── Print summary to console ─────────────────────────────────────────
    print()
    print("  ── Wildfire Alert Check ─────────────────────────────────")
    print(f"  Category column : {cat_col}")
    print(f"  Total rows      : {len(df):,}")
    print(f"  Wildfire Risk   : {len(wildfire_df):,} hotspot(s)", end="")
    print(f" (FRP >= {threshold} MW filter)" if threshold > 0 else "")
    print()

    return send_alert_email(wildfire_df, threshold=threshold, dry_run=dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send Wildfire Risk email alert from features_labeled.csv"
    )
    parser.add_argument(
        "--frp",
        type=float,
        default=None,
        help="Only alert if FRP >= this value (MW). Overrides .env setting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the email in the terminal without actually sending it.",
    )
    args = parser.parse_args()

    ok = send_alerts(frp_threshold=args.frp, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)
