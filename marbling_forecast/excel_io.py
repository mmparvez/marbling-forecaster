"""Write forecasts back into a copy of the workbook in a single pass."""
import pandas as pd
from openpyxl import load_workbook

from .config import DATE_COL, GRADES


def write_forecasts(src, dst, sheet, backtest: pd.DataFrame, future: pd.DataFrame):
    wb = load_workbook(src)
    ws = wb[sheet]
    header = {c.value: c.column for c in ws[1] if c.value}
    date_col = header[DATE_COL]
    rows = {pd.Timestamp(ws.cell(r, date_col).value): r for r in range(2, ws.max_row + 1)
            if ws.cell(r, date_col).value is not None}
    out_col = {g: header.get(f"{g}_fcast%") or ws.max_column + 1 + i for i, g in enumerate(GRADES)}
    for g in GRADES:
        ws.cell(1, out_col[g], f"{g}_fcast%")
    last = max(rows.values())
    for d, r in backtest.iterrows():
        for g in GRADES:
            ws.cell(rows[d], out_col[g], float(r[g]))
    for k, (d, r) in enumerate(future.iterrows(), start=1):
        ws.cell(last + k, date_col, d.to_pydatetime())
        for g in GRADES:
            ws.cell(last + k, out_col[g], float(r[g]))
    wb.save(dst)
