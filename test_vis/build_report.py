"""把 data/measure、data/validate、config/payload_*.json 汇总成一个自包含的 HTML 报告。

用法：
    python test_vis/build_report.py                 # → test_vis/report.html
    python test_vis/build_report.py --out /tmp/x.html --real-only

报告内容：当前生效负载参数 / 每次测量会话的求解质量（m±σ、质心±σ、α、条件数、残差、折算下垂）/
每次到位验证的三方案对比（关节误差、末端误差、耗时、超时数）+ 逐姿态明细 + 自动生成的观察结论。
数据内嵌在 HTML 里，浏览器直接打开，不需要服务端、不需要外网。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEASURE = ROOT / "data" / "measure"
VALIDATE = ROOT / "data" / "validate"
CONFIG = ROOT / "config"
JOINTS = ["肩pitch", "肩roll", "肩yaw", "肘", "腕roll", "腕pitch", "腕yaw"]
MODE_ZH = {"baseline": "基线（现状）", "payload": "负载补偿", "payload_correct": "补偿+修正"}


def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def collect(real_only: bool) -> dict:
    applied = {arm: _read(CONFIG / f"payload_{arm}.json") for arm in ("left", "right")}

    sessions = []
    if MEASURE.exists():
        for d in sorted(MEASURE.iterdir()):
            meta = _read(d / "meta.json")
            if not meta or (real_only and meta.get("controller") == "sim"):
                continue
            samples = _read(d / "samples.json") or []
            solves = []
            for f in sorted(d.glob("payload_*.json")):
                s = _read(f)
                if not s:
                    continue
                solves.append({k: s.get(k) for k in (
                    "source", "fit_alpha", "n_samples", "mass_kg", "mass_std_kg", "com_m", "com_std_m", "alpha", "alpha_std",
                    "regressor_cond", "residual_rms_before_nm", "residual_rms_after_nm", "residual_per_joint_before_nm",
                    "residual_per_joint_after_nm", "sag_est_before_deg", "sag_est_after_deg", "solved_at", "sim_error")}
                              | {"file": f.name, "per_sample_rms_nm": s.get("per_sample_rms_nm") or {}})
            sessions.append({
                "session": d.name, "arm": meta.get("arm"), "controller": meta.get("controller"),
                "poseset": meta.get("poseset"), "created": meta.get("created"), "n_samples": len(samples),
                "n_poses": len({x.get("name") for x in samples}), "opts": meta.get("opts"),
                "payload_during_measure": meta.get("payload_during_measure"), "sim_truth": meta.get("sim_truth"),
                "samples": [{k: x.get(k) for k in ("name", "approach", "settle_s", "timed_out", "err_max_deg")} for x in samples],
                "solves": solves,
                "is_applied_source": any((applied.get(meta.get("arm")) or {}).get("source_session") == d.name for _ in [0]),
            })

    validations = []
    if VALIDATE.exists():
        for f in sorted(VALIDATE.glob("*.json")):
            v = _read(f)
            if not v or (real_only and v.get("controller") == "sim"):
                continue
            modes = {}
            for m in v.get("modes", []):
                rows = [r for r in v.get("rows", []) if r.get("mode") == m]
                if not rows:
                    continue
                modes[m] = {
                    "n": len(rows),
                    "err_max_mean": st.mean(r["err_max_deg"] for r in rows),
                    "err_max_worst": max(r["err_max_deg"] for r in rows),
                    "err_rms_mean": st.mean(r["err_rms_deg"] for r in rows),
                    "ee_mm_mean": st.mean(r["ee_mm"] for r in rows),
                    "ee_mm_worst": max(r["ee_mm"] for r in rows),
                    "traj_s_mean": st.mean(r["traj_s"] for r in rows),
                    "settle_s_mean": st.mean(r["settle_s"] for r in rows),
                    "total_s_mean": st.mean(r["total_s"] for r in rows),
                    "timed_out": sum(1 for r in rows if r.get("timed_out")),
                    "per_joint_abs_mean": [st.mean(abs(r["err_deg"][j]) for r in rows) for j in range(7)] if rows[0].get("err_deg") else None,
                }
            validations.append({
                "file": f.name, "session": v.get("session"), "arm": v.get("arm"), "controller": v.get("controller"),
                "poseset": v.get("poseset"), "payload": v.get("payload"), "finished": v.get("finished"),
                "modes_order": v.get("modes", []), "modes": modes, "sim_truth": v.get("sim_truth"),
                "rows": [{k: r.get(k) for k in ("mode", "pose", "err_max_deg", "err_rms_deg", "ee_mm", "ee_rot_deg",
                                                 "traj_s", "settle_s", "total_s", "timed_out", "err_deg", "corrections")}
                         for r in v.get("rows", [])],
            })

    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(ROOT), "applied": applied,
            "sessions": sessions, "validations": validations, "joints": JOINTS, "mode_zh": MODE_ZH}


def findings(data: dict) -> list[str]:
    out = []
    for s in data["sessions"]:
        for sv in s["solves"]:
            if not sv.get("fit_alpha") or sv.get("mass_kg") is None:
                continue
            cond = sv.get("regressor_cond") or 0
            if sv["mass_kg"] <= 0:
                out.append(f"{s['session']}（{s['arm']}）勾 α 的解病态：m={sv['mass_kg']:.3f} kg（负质量），条件数 {cond:.1f}。应以不勾 α 的解为准。")
            elif cond > 15:
                out.append(f"{s['session']}（{s['arm']}）勾 α 后条件数 {cond:.1f}（不勾时 ≈2.6）：α 与 m 强相关，m 的 σ 也随之放大；"
                           f"两组参数应对照验证后再选。")
    for v in data["validations"]:
        m = v["modes"]
        if "baseline" in m and "payload" in m:
            r = m["payload"]["err_max_mean"] / max(m["baseline"]["err_max_mean"], 1e-9)
            verdict = "达到验收标准（≤ 基线 1/3）" if r <= 1 / 3 else "未达到验收标准（≤ 基线 1/3）"
            out.append(f"{v['file']}（{v['arm']}）：补偿后平均最大关节误差 {m['baseline']['err_max_mean']:.2f}° → {m['payload']['err_max_mean']:.2f}°"
                       f"（{r * 100:.0f}%），末端 {m['baseline']['ee_mm_mean']:.1f} → {m['payload']['ee_mm_mean']:.1f} mm，"
                       f"轨迹时长 {m['baseline']['traj_s_mean']:.1f} vs {m['payload']['traj_s_mean']:.1f} s；{verdict}。")
        if "payload_correct" in m and "payload" in m:
            out.append(f"{v['file']}：再加修正 → {m['payload_correct']['err_max_mean']:.2f}°/{m['payload_correct']['ee_mm_mean']:.1f} mm，"
                       f"代价合计时长 {m['payload']['total_s_mean']:.1f} → {m['payload_correct']['total_s_mean']:.1f} s。")
        total = sum(x["n"] for x in m.values())
        to = sum(x["timed_out"] for x in m.values())
        if total and to == total:
            out.append(f"{v['file']}：全部 {total} 条静止判定都超时（settle ≈ {next(iter(m.values()))['settle_s_mean']:.1f} s 即超时上限），"
                       f"是按超时时刻取的读数。真机 dq 噪声比仿真大，建议放宽静止阈值（速度阈/漂移阈），否则每个姿态白等 8 s。")
    # 同一臂两次只跑 payload_correct 的对比（比如 α=1 与 拟合 α 两组参数）
    by_arm: dict[str, list] = {}
    for v in data["validations"]:
        if v["controller"] != "sim" and list(v["modes"]) == ["payload_correct"]:
            by_arm.setdefault(v["arm"], []).append(v)
    for arm, vs in by_arm.items():
        if len(vs) >= 2:
            desc = "；".join(f"{x['file'][:15]} m={x['payload']['mass_kg']:.3f} α={x['payload']['alpha']:.3f} → "
                           f"修正前第 0 轮平均 {st.mean(r['corrections'][0]['err_max_deg'] for r in x['rows'] if r.get('corrections')):.2f}°，"
                           f"最终 {x['modes']['payload_correct']['err_max_mean']:.2f}°" for x in vs)
            out.append(f"{arm} 臂多组参数对照（仅补偿+修正）：{desc}。")
    return out


HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>末端负载重力补偿 — 测试报告</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1a1a;--mute:#6b7280;--line:#e5e7eb;--ok:#2e9d5b;--warn:#c98a00;--bad:#d64545;--blue:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:16px 32px;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
header h1{margin:0;font-size:18px}header .mute{color:var(--mute);font-size:12px}
main{max-width:1280px;margin:0 auto;padding:24px 32px}
h2{font-size:16px;margin:32px 0 12px}h3{font-size:14px;margin:16px 0 8px;color:#374151}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--mute);font-weight:500;background:#fafafa}td:first-child,th:first-child{text-align:left}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;background:#f0f0f0;color:#374151;margin-right:4px}
.tag.ok{background:#e6f4ec;color:#2e7d4c}.tag.warn{background:#fff4d6;color:#8a5a00}.tag.bad{background:#fbe9e9;color:#b23b3b}.tag.blue{background:#e6eefc;color:#1d4ed8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.bar{position:relative;height:18px;background:#eef1f5;border-radius:4px;overflow:hidden}
.bar>i{position:absolute;left:0;top:0;bottom:0;background:var(--blue);opacity:.85}.bar>b{position:absolute;right:6px;top:0;font-size:11px;font-weight:500;line-height:18px}
.bars{display:grid;grid-template-columns:120px 1fr;gap:6px 10px;align-items:center;font-size:12px}
details{margin-top:8px}summary{cursor:pointer;color:var(--blue);font-size:13px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:13px}.kv div:nth-child(odd){color:var(--mute)}
.small{font-size:12px;color:var(--mute)}ul.find li{margin:4px 0}
.sparks{display:flex;gap:2px;align-items:flex-end;height:28px}.sparks i{display:block;width:6px;background:#93c5fd;border-radius:1px}
.sparks i.to{background:#f59e0b}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
</style></head><body>
<header><h1>末端负载重力补偿 · 测试报告</h1><span class="mute" id="meta"></span></header>
<main>
<section><h2>当前生效参数</h2><div class="grid" id="applied"></div></section>
<section><h2>观察结论（自动生成）</h2><div class="card"><ul class="find" id="findings"></ul></div></section>
<section><h2>到位验证</h2><div id="validations"></div></section>
<section><h2>测量会话与求解质量</h2><div id="sessions"></div></section>
<p class="small" id="foot"></p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const f = (v, n = 2) => (v === null || v === undefined || Number.isNaN(+v)) ? '—' : (+v).toFixed(n);
const pm = (v, s, n = 3) => s == null ? f(v, n) : `${f(v, n)} ± ${f(s, n)}`;
const arr = (a, n = 3) => Array.isArray(a) ? '[' + a.map(x => f(x, n)).join(', ') + ']' : '—';
const armZh = a => a === 'left' ? '左臂' : a === 'right' ? '右臂' : (a || '?');
const ctrlZh = c => c === 'sim' ? '<span class="tag">仿真</span>' : '<span class="tag blue">真机</span>';
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

document.getElementById('meta').textContent = `生成于 ${D.generated_at} · 数据目录 ${D.root}`;
document.getElementById('foot').textContent = `重新生成：python test_vis/build_report.py（数据来自 data/measure、data/validate、config/payload_*.json）`;

// ---- 生效参数
document.getElementById('applied').innerHTML = ['right', 'left'].map(arm => {
  const p = D.applied[arm];
  if (!p) return `<div class="card"><h3>${armZh(arm)}</h3><span class="small">尚未应用负载参数（config/payload_${arm}.json 不存在）</span></div>`;
  return `<div class="card"><h3>${armZh(arm)} <span class="tag ok">生效中</span></h3><div class="kv">
    <div>质量</div><div><b>${f(p.mass_kg, 3)} kg</b></div>
    <div>质心（末端连杆系）</div><div class="mono">${arr(p.com_m, 4)} m</div>
    <div>α（臂自重比例）</div><div>${f(p.alpha, 3)}</div>
    <div>应用时间</div><div>${esc(p.applied_at)}</div>
    <div>来源会话</div><div class="mono">${esc(p.source_session)}</div></div></div>`;
}).join('');

// ---- 结论
document.getElementById('findings').innerHTML = (D.findings || []).map(x => `<li>${esc(x)}</li>`).join('') || '<li class="small">无</li>';

// ---- 验证
const MODES = ['baseline', 'payload', 'payload_correct'];
function bars(modes, key, unit, n) {
  const ms = MODES.filter(m => modes[m]); const mx = Math.max(...ms.map(m => modes[m][key]), 1e-9);
  return `<div class="bars">` + ms.map(m => `<span>${D.mode_zh[m]}</span><div class="bar"><i style="width:${(modes[m][key] / mx * 100).toFixed(1)}%"></i><b>${f(modes[m][key], n)} ${unit}</b></div>`).join('') + `</div>`;
}
document.getElementById('validations').innerHTML = [...D.validations].reverse().map(v => {
  const ms = MODES.filter(m => v.modes[m]);
  const head = `<h3>${esc(v.file)} &nbsp;${ctrlZh(v.controller)} <span class="tag">${armZh(v.arm)}</span> <span class="tag">姿态集 ${esc(v.poseset)}</span>
    ${v.payload ? `<span class="tag">m ${f(v.payload.mass_kg, 3)} kg · α ${f(v.payload.alpha, 3)} · 质心 ${arr(v.payload.com_m, 3)}</span>` : ''}
    ${v.sim_truth ? `<span class="tag warn">仿真真值 m ${f(v.sim_truth.payload_mass, 3)} kg</span>` : ''}
    <span class="small">完成 ${esc(v.finished)}</span></h3>`;
  const table = `<table><thead><tr><th>方案</th><th>姿态数</th><th>最大关节误差 均值</th><th>最差</th><th>RMS 均值</th><th>末端误差 均值</th><th>最差</th><th>轨迹时长</th><th>静止时长</th><th>合计</th><th>静止超时</th></tr></thead><tbody>` +
    ms.map(m => { const x = v.modes[m]; return `<tr><td>${D.mode_zh[m]}</td><td>${x.n}</td><td><b>${f(x.err_max_mean)}°</b></td><td>${f(x.err_max_worst)}°</td><td>${f(x.err_rms_mean)}°</td><td><b>${f(x.ee_mm_mean, 1)} mm</b></td><td>${f(x.ee_mm_worst, 1)} mm</td><td>${f(x.traj_s_mean, 1)} s</td><td>${f(x.settle_s_mean, 1)} s</td><td>${f(x.total_s_mean, 1)} s</td><td>${x.timed_out}/${x.n}${x.timed_out === x.n ? ' <span class="tag warn">全部</span>' : ''}</td></tr>`; }).join('') + `</tbody></table>`;
  const charts = ms.length > 1 ? `<div class="grid" style="margin-top:12px"><div><div class="small">最大关节误差（均值）</div>${bars(v.modes, 'err_max_mean', '°', 2)}</div><div><div class="small">末端位置误差（均值）</div>${bars(v.modes, 'ee_mm_mean', 'mm', 1)}</div><div><div class="small">合计耗时（均值）</div>${bars(v.modes, 'total_s_mean', 's', 1)}</div></div>` : '';
  const pj = ms.some(m => v.modes[m].per_joint_abs_mean) ? `<details><summary>逐关节 |误差| 均值</summary><table><thead><tr><th>方案</th>${D.joints.map(j => `<th>${j}</th>`).join('')}</tr></thead><tbody>` +
    ms.map(m => `<tr><td>${D.mode_zh[m]}</td>${(v.modes[m].per_joint_abs_mean || []).map(x => `<td>${f(x)}°</td>`).join('')}</tr>`).join('') + `</tbody></table></details>` : '';
  const rows = `<details><summary>逐姿态明细（${v.rows.length} 条）</summary><table><thead><tr><th>方案</th><th>姿态</th><th>最大误差</th><th>RMS</th><th>末端</th><th>末端转角</th><th>轨迹</th><th>静止</th><th>合计</th><th>超时</th><th>修正轮次（最大误差）</th><th>逐关节误差 °</th></tr></thead><tbody>` +
    v.rows.map(r => `<tr><td>${D.mode_zh[r.mode] || r.mode}</td><td class="mono">${esc(r.pose)}</td><td>${f(r.err_max_deg)}°</td><td>${f(r.err_rms_deg)}°</td><td>${f(r.ee_mm, 1)} mm</td><td>${f(r.ee_rot_deg)}°</td><td>${f(r.traj_s, 1)}</td><td>${f(r.settle_s, 1)}</td><td>${f(r.total_s, 1)}</td><td>${r.timed_out ? '是' : ''}</td><td class="mono">${(r.corrections || []).map(c => f(c.err_max_deg)).join(' → ') || '—'}</td><td class="mono">${arr(r.err_deg, 2)}</td></tr>`).join('') + `</tbody></table></details>`;
  return `<div class="card">${head}${table}${charts}${pj}${rows}</div>`;
}).join('') || '<div class="card small">没有验证记录</div>';

// ---- 测量会话
document.getElementById('sessions').innerHTML = [...D.sessions].reverse().map(s => {
  const head = `<h3><span class="mono">${esc(s.session)}</span> &nbsp;${ctrlZh(s.controller)} <span class="tag">${armZh(s.arm)}</span> <span class="tag">姿态集 ${esc(s.poseset)}</span>
    <span class="tag">${s.n_poses} 姿态 × ${s.opts && s.opts.bidirectional ? '双向' : '单向'} = ${s.n_samples} 样本</span>
    ${s.is_applied_source ? '<span class="tag ok">当前生效参数来源</span>' : ''}
    ${s.sim_truth ? `<span class="tag warn">仿真真值 m ${f(s.sim_truth.payload_mass, 3)} kg @ ${arr(s.sim_truth.payload_com, 3)}</span>` : ''}
    <span class="small">${esc(s.created)}</span></h3>`;
  const solves = s.solves.length ? `<table><thead><tr><th>解</th><th>力矩源</th><th>拟合 α</th><th>m (kg)</th><th>质心 x/y/z (m)</th><th>质心 σ (mm)</th><th>α</th><th>条件数</th><th>残差 RMS 前→后 (Nm)</th><th>折算下垂 前→后</th><th>求解时间</th></tr></thead><tbody>` +
    s.solves.map(sv => {
      const bad = sv.fit_alpha && sv.mass_kg <= 0 ? '<span class="tag bad">病态（负质量）</span>' : (sv.fit_alpha && (sv.regressor_cond || 0) > 15 ? '<span class="tag warn">条件数高</span>' : '');
      const isApplied = s.is_applied_source && D.applied[s.arm] && Math.abs(D.applied[s.arm].mass_kg - sv.mass_kg) < 5e-4;
      return `<tr><td class="mono">${esc(sv.file)} ${isApplied ? '<span class="tag ok">已应用</span>' : ''}${bad}</td><td>${sv.source === 'pd' ? 'PD 重构' : 'tau_est'}</td><td>${sv.fit_alpha ? '是' : '否'}</td>
        <td><b>${pm(sv.mass_kg, sv.mass_std_kg)}</b></td><td class="mono">${arr(sv.com_m, 4)}</td><td class="mono">${Array.isArray(sv.com_std_m) ? arr(sv.com_std_m.map(x => x * 1000), 1) : '—'}</td>
        <td>${pm(sv.alpha, sv.alpha_std)}</td><td>${f(sv.regressor_cond, 1)}</td><td>${f(sv.residual_rms_before_nm)} → <b>${f(sv.residual_rms_after_nm)}</b></td><td>${f(sv.sag_est_before_deg)}° → <b>${f(sv.sag_est_after_deg)}°</b></td><td class="small">${esc(sv.solved_at)}</td></tr>` +
        (sv.sim_error ? `<tr><td colspan="11" class="small">对照仿真真值：Δm ${f(sv.sim_error.mass_kg, 3)} kg，Δ质心 ${arr(sv.sim_error.com_mm, 1)} mm</td></tr>` : '');
    }).join('') + `</tbody></table>` : '<div class="small">这次会话没有求解记录</div>';
  const pj = s.solves.filter(sv => sv.residual_per_joint_after_nm).map(sv => `<tr><td class="mono">${esc(sv.file)}</td>${sv.residual_per_joint_before_nm.map((b, j) => `<td>${f(b)} → ${f(sv.residual_per_joint_after_nm[j])}</td>`).join('')}</tr>`).join('');
  const pjT = pj ? `<details><summary>逐关节残差 RMS 前→后 (Nm)</summary><table><thead><tr><th>解</th>${D.joints.map(j => `<th>${j}</th>`).join('')}</tr></thead><tbody>${pj}</tbody></table></details>` : '';
  const to = s.samples.filter(x => x.timed_out).length;
  const samples = `<details><summary>采样明细（${s.samples.length} 条，静止超时 ${to}）</summary><table><thead><tr><th>姿态</th><th>方向</th><th>静止耗时</th><th>超时</th><th>到位误差</th>${s.solves.map(sv => `<th>残差 RMS ${esc(sv.file.replace('payload_', '').replace('.json', ''))}</th>`).join('')}</tr></thead><tbody>` +
    s.samples.map(x => { const key = (x.name || '') + (x.approach || ''); return `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.approach)}</td><td>${f(x.settle_s, 1)} s</td><td>${x.timed_out ? '是' : ''}</td><td>${f(x.err_max_deg)}°</td>${s.solves.map(sv => `<td>${f(sv.per_sample_rms_nm[key])}</td>`).join('')}</tr>`; }).join('') + `</tbody></table></details>`;
  return `<div class="card">${head}${solves}${pjT}${samples}</div>`;
}).join('') || '<div class="card small">没有测量会话</div>';
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成末端负载重力补偿测试报告 HTML")
    ap.add_argument("--out", default=str(HERE / "report.html"))
    ap.add_argument("--real-only", action="store_true", help="只包含真机（h2）的会话/验证，不含仿真")
    a = ap.parse_args()
    data = collect(a.real_only)
    data["findings"] = findings(data)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    print(f"[report] {len(data['sessions'])} 个测量会话，{len(data['validations'])} 次验证 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
