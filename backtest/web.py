"""
Small READ-ONLY dashboard: live paper positions + saved backtest reports.

  * GET only - no route writes to the DB, the reports dir, or any broker.
  * Positions come from the latest portfolio snapshot the (paper) engine
    stored via `DatabaseManager`; reports come from JSON files written by
    `python -m backtest`.
  * No credentials are read or required.

Run:  uvicorn --factory backtest.web:create_app   (env BACKTEST_REPORTS_DIR,
      STOCKBOT_DB_URL optional)  or  create_app(reports_dir=..., db_url=...)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from backtest.report import list_report_files
from database.database import DatabaseManager

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StockBot Backtests</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;padding:16px;background:#fff;color:#111;max-width:960px;margin-inline:auto}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}th,td{border-color:#333!important}}
h1{font-size:20px}h2{font-size:16px;margin-top:24px}table{border-collapse:collapse;width:100%}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}
.pos{color:#1a7f37}.neg{color:#cf222e}svg{width:100%;height:160px;border:1px solid #8884}
.note{opacity:.7}
</style></head><body>
<h1>StockBot &mdash; paper trading &amp; backtests</h1>
<p class="note">Read-only view. Paper simulation only.</p>
<h2>Open positions</h2><div id="pos">loading&hellip;</div>
<h2>Backtest runs</h2><div id="runs">loading&hellip;</div>
<div id="detail"></div>
<script>
const $=id=>document.getElementById(id);
const cls=v=>v>0?'pos':v<0?'neg':'';
const money=v=>'$'+v.toLocaleString(undefined,{maximumFractionDigits:2});
const pct=v=>(v*100).toFixed(2)+'%';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function table(h,rows){return '<table><tr>'+h.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.join('')+'</tr>').join('')+'</table>'}
async function load(){
 const p=await (await fetch('api/positions')).json();
 const ks=Object.keys(p.positions);
 $('pos').innerHTML=ks.length?table(['Symbol','Qty','Avg entry','Mkt value','Unrealized'],
  ks.map(k=>{const x=p.positions[k];return ['<td>'+esc(k)+'</td>','<td>'+x.quantity+'</td>','<td>'+money(x.avg_entry_price)+'</td>','<td>'+money(x.market_value)+'</td>','<td class="'+cls(x.unrealized_pnl)+'">'+money(x.unrealized_pnl)+'</td>']})):
  '<p class="note">No open positions'+(p.timestamp?'':' (no snapshot stored yet)')+'.</p>';
 const r=await (await fetch('api/reports')).json();
 $('runs').innerHTML=r.length?table(['Run','Strategy','P&L','Return','Buy&hold','Max DD','Win rate','Trades'],
  r.map(x=>['<td><a href="#" data-n="'+esc(x.id)+'">'+esc(x.name)+'</a></td>','<td>'+esc(x.strategy)+'</td>','<td class="'+cls(x.total_pnl)+'">'+money(x.total_pnl)+'</td>','<td>'+pct(x.total_return_pct)+'</td>','<td>'+pct(x.benchmark_return_pct)+'</td>','<td>'+pct(x.max_drawdown_pct)+'</td>','<td>'+pct(x.win_rate)+'</td>','<td>'+x.num_trades+'</td>'])):
  '<p class="note">No reports yet. Run <code>python -m backtest --synthetic</code>.</p>';
 document.querySelectorAll('a[data-n]').forEach(a=>a.onclick=e=>{e.preventDefault();show(a.dataset.n)});
}
async function show(id){
 const d=await (await fetch('api/reports/'+encodeURIComponent(id))).json();
 const c=d.equity_curve.map(x=>x[1]);let svg='';
 if(c.length>1){const lo=Math.min(...c),hi=Math.max(...c),w=600,h=150;
  const pts=c.map((v,i)=>(i/(c.length-1)*w).toFixed(1)+','+(h-(v-lo)/((hi-lo)||1)*h).toFixed(1)).join(' ');
  svg='<svg viewBox="0 0 '+w+' '+h+'" preserveAspectRatio="none" role="img" aria-label="Equity curve"><polyline fill="none" stroke="currentColor" stroke-width="1.5" points="'+pts+'"/></svg>'}
 $('detail').innerHTML='<h2>'+esc(d.name)+'</h2>'+svg+table(['Series','P&L','Return','Max DD','Win rate','Trades'],
  d.per_symbol.map(s=>['<td>'+esc(s.label)+'</td>','<td class="'+cls(s.pnl)+'">'+money(s.pnl)+'</td>','<td>'+pct(s.return_pct)+'</td>','<td>'+pct(s.max_drawdown_pct)+'</td>','<td>'+pct(s.win_rate)+'</td>','<td>'+s.num_trades+'</td>']));
}
load();
</script></body></html>"""

_SUMMARY_KEYS = (
    "name", "strategy", "created_at", "total_pnl", "total_return_pct",
    "benchmark_return_pct", "max_drawdown_pct", "win_rate", "num_trades",
    "profit_factor", "sharpe_ratio",
)


def create_app(reports_dir: Optional[str] = None, db_url: Optional[str] = None) -> FastAPI:
    reports_dir = reports_dir or os.environ.get("BACKTEST_REPORTS_DIR", "backtest_reports")
    db_url = db_url or os.environ.get("STOCKBOT_DB_URL", "sqlite:///./stockbot.db")
    db = DatabaseManager(db_url)
    db.init_db()  # creates tables if absent; the dashboard itself never writes rows

    app = FastAPI(title="StockBot Backtest Dashboard")

    def _files() -> dict:
        # Only files that actually exist in the reports dir are addressable,
        # so a request id can never escape the directory.
        return {p.stem: p for p in list_report_files(reports_dir)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/positions")
    def positions():
        snap = db.get_latest_portfolio_snapshot()
        if snap is None:
            return {"timestamp": None, "cash": None, "total_value": None, "positions": {}}
        return {k: snap[k] for k in ("timestamp", "cash", "total_value", "positions")}

    @app.get("/api/reports")
    def reports():
        out = []
        for rid, path in _files().items():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append({"id": rid, **{k: data.get(k) for k in _SUMMARY_KEYS}})
            except (OSError, ValueError):
                continue  # skip unreadable/corrupt files
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    @app.get("/api/reports/{report_id}")
    def report(report_id: str):
        path = _files().get(report_id)
        if path is None:
            raise HTTPException(status_code=404, detail="report not found")
        return json.loads(path.read_text(encoding="utf-8"))

    return app
