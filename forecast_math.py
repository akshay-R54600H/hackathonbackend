from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Any

import pandas as pd


@dataclass
class Assumptions:
    months: int
    initial_salespeople: int
    salespeople_added_per_month: int
    deals_per_salesperson: float
    large_customer_revenue_per_month: float
    marketing_spend_per_month: float
    average_cac: float
    conversion_rate: float
    sme_customer_revenue_per_month: float
    overrides: List[Dict] = None


def _to_number(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.startswith("MISSING_"):
            return float(default)
        s = value.replace(",", "").replace("$", "").strip()
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                return float(default)
        try:
            return float(s)
        except ValueError:
            return float(default)
    return float(default)


def apply_overrides_for_month(month: int, field: str, base_value: float, overrides: List[Dict]) -> float:
    """Apply all overrides for a specific month and field."""
    if not overrides:
        return base_value
    
    # Collect all overrides for this month and field
    month_overrides = [o for o in overrides if o.get("field") == field and o.get("month") == month]
    
    current_value = base_value
    
    # Apply overrides in order
    for o in month_overrides:
        value = _to_number(o.get("value", 0), 0)
        op = str(o.get("op", "add")).lower()
        if op == "set":
            current_value = value
        elif op == "add":
            current_value += value
        elif op == "multiply":
            current_value *= value
    
    return current_value

def build_forecast_df(assumptions: Assumptions) -> pd.DataFrame:
    months = int(assumptions.months)
    month_index = list(range(1, months + 1))
    month_label = [f"M{i}" for i in month_index]

    # Calculate salespeople progression with overrides
    monthly_increment = _to_number(assumptions.salespeople_added_per_month, 0)
    initial_salespeople = _to_number(assumptions.initial_salespeople, 0)
    
    salespeople = []
    current_value = int(initial_salespeople)

    for i, m in enumerate(month_index, start=1):
        # Normal monthly increment
        if i == 1:
            current_value = int(initial_salespeople)
        else:
            current_value += int(monthly_increment)

        # Apply overrides for this month
        current_value = apply_overrides_for_month(i, "salespeople", current_value, assumptions.overrides or [])
        salespeople.append(int(current_value))

    new_large_customers = [sp * assumptions.deals_per_salesperson for sp in salespeople]

    cumulative_large_customers: List[float] = []
    total_lc = 0.0
    for added in new_large_customers:
        total_lc += added
        cumulative_large_customers.append(total_lc)

    # Calculate revenue with potential overrides
    revenue_large = []
    for i, clc in enumerate(cumulative_large_customers):
        base_revenue = clc * assumptions.large_customer_revenue_per_month
        # Apply overrides for large customer revenue
        adjusted_revenue = apply_overrides_for_month(i + 1, "large_customer_revenue_per_month", base_revenue, assumptions.overrides or [])
        revenue_large.append(adjusted_revenue)

    # Calculate demo leads and SME customers with overrides
    demo_leads = []
    new_sme_customers = []
    
    for i, _ in enumerate(month_index):
        # Apply overrides for marketing spend and CAC
        marketing_spend = apply_overrides_for_month(i + 1, "marketing_spend_per_month", assumptions.marketing_spend_per_month, assumptions.overrides or [])
        cac = apply_overrides_for_month(i + 1, "average_cac", assumptions.average_cac, assumptions.overrides or [])
        
        # Calculate leads
        leads = marketing_spend / cac if cac > 0 else 0
        demo_leads.append(leads)
        
        # Apply overrides for conversion rate
        conversion_rate = apply_overrides_for_month(i + 1, "conversion_rate", assumptions.conversion_rate, assumptions.overrides or [])
        new_sme = leads * conversion_rate
        new_sme_customers.append(new_sme)

    cumulative_sme_customers: List[float] = []
    total_sme = 0.0
    for added in new_sme_customers:
        total_sme += added
        cumulative_sme_customers.append(total_sme)

    # Calculate SME revenue with overrides
    revenue_sme = []
    for i, csc in enumerate(cumulative_sme_customers):
        base_revenue = csc * assumptions.sme_customer_revenue_per_month
        # Apply overrides for SME customer revenue
        adjusted_revenue = apply_overrides_for_month(i + 1, "sme_customer_revenue_per_month", base_revenue, assumptions.overrides or [])
        revenue_sme.append(adjusted_revenue)

    total_revenue = [rl + rs for rl, rs in zip(revenue_large, revenue_sme)]
    total_revenue_mn = [tr / 1_000_000 for tr in total_revenue]
    
    # Calculate new paying customers onboarded per month (SME + Large)
    new_paying_customers = [nl + ns for nl, ns in zip(new_large_customers, new_sme_customers)]
    
    # Calculate total cumulative paying customers (SME + Large)
    cumulative_paying_customers = [lc + sc for lc, sc in zip(cumulative_large_customers, cumulative_sme_customers)]

    df = pd.DataFrame(
        {
            "month_index": month_index,
            "month_label": month_label,
            "salespeople": salespeople,
            "new_large_customers": new_large_customers,
            "cumulative_large_customers": cumulative_large_customers,
            "revenue_large": revenue_large,
            "demo_leads": demo_leads,
            "new_sme_customers": new_sme_customers,
            "cumulative_sme_customers": cumulative_sme_customers,
            "revenue_sme": revenue_sme,
            "new_paying_customers": new_paying_customers,
            "cumulative_paying_customers": cumulative_paying_customers,
            "total_revenue": total_revenue,
            "total_revenue_mn": total_revenue_mn,
        }
    )

    return df


def df_to_json_table(df: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "month": str(row["month_label"]),
                "salespeople": int(row["salespeople"]),
                "large_customers": int(row["cumulative_large_customers"]),
                "revenue_large": round(float(row["revenue_large"])),
                "sme_customers": int(row["cumulative_sme_customers"]),
                "revenue_sme": round(float(row["revenue_sme"])),
                "new_paying_customers": int(row["new_paying_customers"]),
                "total_paying_customers": int(row["cumulative_paying_customers"]),
                "total_revenue": round(float(row["total_revenue"])),
                "total_revenue_mn": round(float(row["total_revenue_mn"]), 2),
            }
        )
    return records


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Forecast") -> bytes:
    output = BytesIO()
    excel_df = df[
        [
            "month_label",
            "salespeople",
            "new_large_customers",
            "cumulative_large_customers",
            "revenue_large",
            "demo_leads",
            "new_sme_customers",
            "cumulative_sme_customers",
            "revenue_sme",
            "new_paying_customers",
            "cumulative_paying_customers",
            "total_revenue",
            "total_revenue_mn",
        ]
    ].rename(
        columns={
            "month_label": "Month",
            "salespeople": "Salespeople",
            "new_large_customers": "New Large Customers",
            "cumulative_large_customers": "Cumulative Large Customers",
            "revenue_large": "Revenue Large",
            "demo_leads": "Demo Leads",
            "new_sme_customers": "New SME Customers",
            "cumulative_sme_customers": "Cumulative SME Customers",
            "revenue_sme": "Revenue SME",
            "new_paying_customers": "New Paying Customers Onboarded",
            "cumulative_paying_customers": "Total Paying Customers",
            "total_revenue": "Total Revenue",
            "total_revenue_mn": "Total Revenue (Mn)",
        }
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        excel_df.to_excel(writer, index=False, sheet_name=sheet_name)

    output.seek(0)
    return output.read()


