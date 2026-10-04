"""Independently verify the submitted Q4 workbooks without any model NPZ files.

Public interface: audit_workbook(workbook_path, attachment_dir, tolerance=1e-6).
No optimization modules are imported and no input workbook is changed.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import re

import numpy as np
import openpyxl


START = dt.date(2025, 2, 1)
DAYS = 334
LOW, HIGH, ETA, LIMIT = 1200., 10800., .9, 5000. / 6


def _rows(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result = {s.title: list(s.values) for s in wb}
    wb.close()
    return result


def _date(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value).split(" ")[0])


def _span(value):
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(\+1)?\s*[-—–]\s*(\d{1,2}):(\d{2})(\+1)?\s*", str(value))
    if not match:
        raise ValueError("Invalid emergency interval: " + str(value))
    h0, m0, n0, h1, m1, n1 = match.groups()
    if int(m0) % 10 or int(m1) % 10:
        raise ValueError("Interval is not on a ten-minute boundary: " + str(value))
    start = int(h0) * 6 + int(m0) // 10 + (144 if n0 else 0)
    end = int(h1) * 6 + int(m1) // 10 + (144 if n1 else 0)
    if end == 0 and start > 0:
        end = 144
    if not 0 <= start < end <= 144:
        raise ValueError("Interval lies outside its daily row: " + str(value))
    return start, end


def _independent_dispatch(q, net, initial_soc):
    c = np.zeros_like(q)
    discharge = np.zeros_like(q)
    emergency = np.zeros_like(q)
    spill = np.zeros_like(q)
    soc = np.empty((DAYS, 145))
    energy = float(initial_soc)
    for day in range(DAYS):
        soc[day, 0] = energy
        for slot in range(144):
            surplus = float(q[day, slot] - net[day, slot])
            if surplus >= 0:
                charge = min(surplus, LIMIT, max(0., (HIGH - energy) / ETA))
                c[day, slot] = charge
                spill[day, slot] = surplus - charge
                energy += ETA * charge
            else:
                used = min(-surplus, LIMIT, max(0., (energy - LOW) * ETA))
                discharge[day, slot] = used
                emergency[day, slot] = -surplus - used
                energy -= used / ETA
            soc[day, slot + 1] = energy
    return {"c": c, "d": discharge, "e": emergency, "w": spill, "s": soc}


def audit_workbook(workbook_path, attachment_dir, tolerance=1e-6):
    workbook_path, attachment_dir = Path(workbook_path), Path(attachment_dir)
    model = "q4-3" if workbook_path.name == "result4-3.xlsx" else "q4-2"
    if workbook_path.name not in ("result4-2.xlsx", "result4-3.xlsx"):
        raise ValueError("Expected result4-2.xlsx or result4-3.xlsx")
    tables = _rows(workbook_path)
    template = _rows(attachment_dir / "附件5" / workbook_path.name)
    if list(tables) != list(template):
        raise ValueError("Worksheet names/order differ from the official template")
    widths = {"计划购电量": 147, "调整购电量": 147, "充放电量": 6, "紧急购电量": 3}
    headers = {name: list(rows[0][:widths[name]]) == list(template[name][0][:widths[name]])
               for name, rows in tables.items()}
    load_tables = list(_rows(attachment_dir / "附件2.xlsx").values())
    price_rows = next(iter(_rows(attachment_dir / "附件4.xlsx").values()))
    expected_dates = [START + dt.timedelta(days=d) for d in range(DAYS)]
    for source in load_tables[:2] + [price_rows]:
        if [_date(r[0]) for r in source[32:366]] != expected_dates:
            raise ValueError("Original attachment formal-period dates differ")
    load = np.asarray([r[1:145] for r in load_tables[0][32:366]], float) / 6
    pv = np.asarray([r[1:145] for r in load_tables[1][32:366]], float) / 6
    price = np.asarray([r[1:145] for r in price_rows[32:366]], float)
    plans = {}
    errors = {}

    def check(name, actual, expected):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(name + ": unexpected shape or missing/non-finite value")
        delta = np.abs(actual - expected)
        errors[name] = {"values_checked": int(delta.size),
                        "maximum_absolute_error": float(delta.max(initial=0)),
                        "values_over_tolerance": int((delta > tolerance).sum())}

    for name in ("计划购电量", "调整购电量"):
        if name not in tables:
            continue
        rows = tables[name][1:DAYS + 1]
        if len(rows) != DAYS or [_date(r[0]) for r in rows] != expected_dates:
            raise ValueError(name + ": expected all 334 dates in order")
        values = np.asarray([r[1:145] for r in rows], float)
        if values.shape != (DAYS, 144) or not np.isfinite(values).all() or values.min() < -tolerance:
            raise ValueError(name + ": invalid procurement values")
        plans[name] = values
        check(name + "_全天购电量", [r[145] for r in rows], values.sum(axis=1))
    q0 = plans["计划购电量"]
    q = plans.get("调整购电量", q0)
    if model == "q4-3":
        check("0点原合同前六小时执行一致", q[:, :36], q0[:, :36])
    storage = tables["充放电量"]
    initial = float(storage[1][5])
    if not LOW - tolerance <= initial <= HIGH + tolerance:
        raise ValueError("The first formal-period initial SOC is outside physical bounds")
    trajectory = _independent_dispatch(q, load - pv, initial)
    for day in range(DAYS):
        if _date(storage[1 + day * 6][0]) != expected_dates[day]:
            raise ValueError("Storage dates are not in order")
        if [r[1] for r in storage[1 + day * 6:7 + day * 6]] != [r[1] for r in template["充放电量"][1:7]]:
            raise ValueError("Storage intervals differ from the six official template intervals")
    for key, col, label in (("c", 2, "四小时充电量"), ("d", 3, "四小时放电量")):
        check(label, [[storage[1 + day * 6 + block][col] for block in range(6)] for day in range(DAYS)],
              trajectory[key].reshape(DAYS, 6, 24).sum(axis=2))
    check("日初SOC", [storage[1 + day * 6][5] for day in range(DAYS)], trajectory["s"][:, 0])
    check("日末SOC", [storage[2 + day * 6][5] for day in range(DAYS)], trajectory["s"][:, -1])
    original_fee = (price * q0).sum(axis=1)
    ordinary_fee = (price * (q + .5 * np.abs(q - q0))).sum(axis=1)
    check("原计划全天费用", [r[146] for r in tables["计划购电量"][1:DAYS + 1]], original_fee)
    if model == "q4-3":
        check("调整后全天费用", [r[146] for r in tables["调整购电量"][1:DAYS + 1]], ordinary_fee)
    coverage = np.zeros((DAYS, 144), dtype=bool)
    emergency_observed, emergency_expected = [], []
    day = None
    for row in tables["紧急购电量"][1:]:
        if all(v is None for v in row):
            continue
        date, span, quantity = row[:3]
        if date is not None:
            day = (_date(date) - START).days
        if day is None or not 0 <= day < DAYS:
            raise ValueError("Emergency row lacks a valid date")
        if span in (None, "无"):
            if abs(float(quantity or 0)) > tolerance:
                raise ValueError("Emergency row has quantity without an interval")
            continue
        first, last = _span(span)
        if coverage[day, first:last].any():
            raise ValueError("Emergency intervals overlap")
        coverage[day, first:last] = True
        emergency_observed.append(float(quantity or 0))
        emergency_expected.append(float(trajectory["e"][day, first:last].sum()))
    check("紧急购电区间电量", emergency_observed, emergency_expected)
    check("未列区间紧急购电应为零", np.where(coverage, 0., trajectory["e"]), np.zeros((DAYS, 144)))
    balance_error = float(np.max(abs(q + trajectory["d"] + trajectory["e"] - (load - pv) - trajectory["c"] - trajectory["w"])))
    emergency_fee = float((5 * price * trajectory["e"]).sum())
    passed = all(headers.values()) and all(v["values_over_tolerance"] == 0 for v in errors.values()) and balance_error <= tolerance
    return {"model": model, "file": workbook_path.name, "status": "PASS" if passed else "FAIL",
            "formal_days": DAYS, "ten_minute_intervals_replayed": DAYS * 144,
            "official_template_headers_unchanged": headers,
            "procurement_columns_used_in_original_order": True,
            "tolerance": tolerance, "comparisons": errors,
            "initial_soc_read_from_original_table": initial,
            "final_soc_replayed": float(trajectory["s"][-1, -1]),
            "minimum_soc": float(trajectory["s"].min()), "maximum_soc": float(trajectory["s"].max()),
            "energy_balance_maximum_error": balance_error,
            "normal_and_adjustment_cost": float(ordinary_fee.sum()),
            "emergency_cost": emergency_fee, "total_cost": float(ordinary_fee.sum()) + emergency_fee,
            "emergency_kwh": float(trajectory["e"].sum()),
            "scope": "Independent physical and billing replay from original Excel plus attachments2/4; no model NPZ, no optimization, no workbook modification",
            "limitation": "The first formal-period SOC is taken from the workbook. This check does not reconstruct the absent January procurement or certify prediction/optimization optimality."}


def main():
    source_root = Path(__file__).resolve().parent.parent
    support_root = source_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=support_root / "01_计算结果")
    parser.add_argument("--attachments", type=Path, default=source_root / "附件")
    parser.add_argument("--report", type=Path, default=support_root / "03_核验记录/Q4原表独立核验.json")
    args = parser.parse_args()
    reports = [audit_workbook(args.results / name, args.attachments) for name in ("result4-2.xlsx", "result4-3.xlsx")]
    result = {"status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL", "workbooks": reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
