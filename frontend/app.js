'use strict';
const $ = (id) => document.getElementById(id);
const fmt = (v, n = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(n) : '—');
const deg = (r) => r * 180 / Math.PI;
const JOINTS = ['sh_pitch', 'sh_roll', 'sh_yaw', 'elbow', 'wr_roll', 'wr_pitch', 'wr_yaw'];

const S = { status: null, lastTaskName: null, solve: null };

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body?.error || `${r.status} ${r.statusText}`);
  return body;
}
const post = (p, body = {}) => api(p, { method: 'POST', body: JSON.stringify(body) });
function msg(t, cls = '') { $('msg').textContent = t; $('msg').className = cls; }
async function act(fn, okText) {
  try { const r = await fn(); if (okText) msg(okText, 'ok'); return r; }
  catch (e) { msg(e.message, 'err'); throw e; }
}

// ---------------------------------------------------------------- 状态
let statusInFlight = false;
async function pollStatus() {
  if (statusInFlight) return; statusInFlight = true;
  try { render(await api('/api/status')); } catch (e) { $('pill').textContent = '服务不可达'; $('pill').className = 'pill bad'; }
  finally { statusInFlight = false; }
}

function render(st) {
  S.status = st;
  const c = st.ctrl;
  // 头部
  const badge = $('mode-badge');
  if (st.mode === 'sim') { badge.textContent = '仿真'; badge.className = 'badge sim'; }
  else if (st.controllable) { badge.textContent = '真机 · 可运动 ⚠'; badge.className = 'badge real'; }
  else { badge.textContent = '真机（只读）'; badge.className = 'badge ro'; }
  if (document.activeElement !== $('mode-select')) $('mode-select').value = st.mode;
  if (document.activeElement !== $('arm-select')) $('arm-select').value = st.arm;
  $('mode-error').textContent = st.mode_error || '';
  $('panel-sim').classList.toggle('hidden', st.mode !== 'sim');
  const engaged = !!(c && c.engaged), jog = !!(c && c.jog_enabled);
  $('engage-state').textContent = c ? (engaged ? `已接管 · 权重 ${fmt(c.weight, 2)} · ${jog ? '点动' : '保持'}` : '未接管') : '无控制器';
  const sdk = st.arm_sdk || {};
  const foreign = !!sdk.foreign;
  $('btn-engage').disabled = !st.controllable || (engaged && jog) || (foreign && !engaged);
  $('btn-release').disabled = !st.controllable || !engaged;
  const a = $('armsdk');
  if (st.mode === 'sim' && !foreign) { a.textContent = ''; }
  else if (foreign) {
    a.innerHTML = `<span class="bad">⚠ rt/arm_sdk 上有其他发布者（约 ${fmt(sdk.foreign_rate_hz, 0)} Hz，权重 ${fmt(sdk.last_weight, 2)}，kp ${fmt(sdk.last_kp, 0)}）${engaged ? ' —— 正在和我们抢臂，请立即释放并停掉对方' : '，禁止接管'}</span>` +
      (sdk.local_hints?.length ? `<br>本机疑似：${sdk.local_hints.map((h) => `pid ${h.pid} <code>${h.cmd.slice(0, 70)}</code>`).join('；')}` : '<br>本机没找到可疑进程，发布者可能在另一台机器上');
  } else { a.innerHTML = `<span class="good">rt/arm_sdk ${engaged ? `只有我们在发（${fmt(sdk.rate_hz, 0)} Hz）` : '空闲，可以接管'}</span>`; }
  const pill = $('pill');
  if (!c) { pill.textContent = '无控制器'; pill.className = 'pill bad'; }
  else if (c.kind === 'h2-readonly') { pill.textContent = c.lowstate_ok ? 'rt/lowstate OK（只读）' : 'rt/lowstate 无数据'; pill.className = 'pill ' + (c.lowstate_ok ? 'warn' : 'bad'); }
  else { pill.textContent = `${c.kind} · 限速 ${fmt(c.max_speed_rad_s, 2)} rad/s`; pill.className = 'pill ok'; }

  renderJoints(c);
  renderPoses(st);
  renderTask(st.task);
  renderSessions(st.sessions);
  renderPayload(st);
  renderValidations(st.validations);
  if (st.sim_truth && document.activeElement?.id?.startsWith('tr-') === false) { /* 不覆盖正在编辑的输入 */ }
}

function renderJoints(c) {
  const tb = $('joints-table').tBodies[0];
  if (!c || !c.measured_rad) { tb.innerHTML = '<tr><td>无数据</td></tr>'; return; }
  const rows = ['<tr><th>关节</th><th>实测 rad</th><th>指令 rad</th><th>偏差 °</th><th>τ_est</th><th>τ_ff</th></tr>'];
  for (let i = 0; i < 7; i++) {
    const q = c.measured_rad[i], qc = c.cmd_rad ? c.cmd_rad[i] : null;
    const e = qc == null ? null : deg(q - qc);
    const cls = e == null ? '' : Math.abs(e) > 1 ? 'bad' : Math.abs(e) > 0.3 ? 'warn' : 'good';
    rows.push(`<tr><td>${JOINTS[i]}</td><td>${fmt(q, 4)}</td><td>${fmt(qc, 4)}</td><td class="${cls}">${fmt(e, 2)}</td>` +
      `<td>${fmt(c.tau_est_nm?.[i], 2)}</td><td>${fmt(c.last_sent_tau_ff_nm?.[i], 2)}</td></tr>`);
  }
  tb.innerHTML = rows.join('');
  const p = c.payload || {};
  $('payload-now').innerHTML = `当前前馈：α=${fmt(p.alpha, 3)} · 负载 <b>${fmt(p.mass_kg, 3)} kg</b> @ [${(p.com_m || []).map((v) => fmt(v, 3)).join(', ')}] (末端连杆系, m)` +
    (c.sim_truth ? ` &nbsp;<span class="warn">仿真真值 ${fmt(c.sim_truth.payload_mass, 3)} kg @ [${c.sim_truth.payload_com.map((v) => fmt(v, 3)).join(', ')}]，臂质量×${fmt(c.sim_truth.arm_mass_scale, 3)}</span>` : '');
}

function renderPoses(st) {
  const p = st.poses;
  const condCls = p?.cond == null ? '' : p.cond > 50 ? 'bad' : p.cond > 10 ? 'warn' : 'good';
  $('poses-meta').innerHTML = p
    ? `<b>${p.name}</b> · ${p.n} 个 · 条件数 <span class="${condCls}">${fmt(p.cond, 1)}</span>${p.note ? ' · ' + p.note : ''}`
    : `（${st.arm} 臂还没有姿态集：新建一个后示教，或自动生成）`;
  const sel = $('poseset-select');
  const html = (st.posesets || []).map((s) => `<option value="${s.name}">${s.name} (${s.n})</option>`).join('');
  if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
  if (p && document.activeElement !== sel) sel.value = p.name;
  const box = $('pose-chips');
  const cur = st.task?.task === 'measure' ? st.task.current : -1;
  const done = new Set((st.task?.samples || []).map((s) => s.index));
  const chips = (p?.poses || []).map((x, i) => `<button data-i="${i}" title="${x.source === 'teach' ? '示教 ' : ''}${x.recorded_at || ''}\n${x.q.map((v) => v.toFixed(2)).join(', ')} → 腕 ${(x.tip_xyz || []).map((v) => v.toFixed(2)).join(',')}" class="${i === cur ? 'cur' : done.has(i) ? 'done' : x.source === 'teach' ? 'teach' : ''}">${x.name}</button>`).join('');
  if (box.innerHTML !== chips) box.innerHTML = chips;
  if (!$('goto-inputs').children.length) buildGotoInputs(p?.home_q);
  renderTeach(st);
}

function renderTeach(st) {
  const t = st.teach || {}, c = st.ctrl;
  const panel = $('panel-teach');
  panel.classList.toggle('active', !!t.active && !t.locked); panel.classList.toggle('locked', !!t.active && !!t.locked);
  $('teach-state').textContent = t.active ? (t.locked ? '🔒 已锁定' : '🖐 卸力中') + (t.message ? ' · ' + t.message : '') : (t.last_pose ? `上次记录 ${t.last_pose.name}` : '');
  const can = st.controllable && !!(c && c.engaged) && !(st.task && st.task.running);
  $('btn-teach-start').disabled = !can || t.active;
  $('btn-teach-toggle').disabled = !t.active;
  $('btn-teach-toggle').textContent = t.active && t.locked ? '空格：继续拖动' : '空格：锁定并记录';
  $('btn-teach-end').disabled = !t.active;
  $('btn-add-current').disabled = !st.poses || !(c && c.measured_rad);
  $('teach-sim').classList.toggle('hidden', st.mode !== 'sim');
  // 阈值输入框：只在用户没聚焦时用服务端值回填
  const s = t.still || {};
  const fill = (id, v) => { const el = $(id); if (document.activeElement !== el && v != null && el.value === '') el.value = v; };
  fill('ts-dq', s.still_dq_rad_s); fill('ts-drift', s.still_drift_rad); fill('ts-win', s.still_window_s); fill('ts-to', s.settle_timeout_s);
  const ls = t.last_still;
  $('teach-still-last').innerHTML = ls ? `上次锁定：${ls.settle_s}s，max|dq|=${fmt(ls.max_dq_rad_s, 4)}，漂移=${fmt(ls.drift_rad, 4)} ` +
    (ls.timed_out ? '<span class="warn">超时（已记录）</span>' : '<span class="good">静止</span>') : '';
  if (st.mode === 'sim' && !$('teach-sliders').children.length && c?.limits_rad) buildTeachSliders(c.limits_rad, c.measured_rad);
  // 示教卸力时把滑块同步到实测（人手没动时）
  if (st.mode === 'sim' && c?.measured_rad && !(t.active && !t.locked && sliderDragging)) {
    document.querySelectorAll('.t-q').forEach((el, i) => { el.value = c.measured_rad[i]; el.nextElementSibling.textContent = c.measured_rad[i].toFixed(2); });
  }
}

let sliderDragging = false, pushTimer = null;
function buildTeachSliders(limits, q) {
  $('teach-sliders').innerHTML = JOINTS.map((n, i) => `<label>${n}<input type="range" class="t-q" data-i="${i}" min="${limits[i][0]}" max="${limits[i][1]}" step="0.005" value="${q ? q[i] : 0}"><span>${q ? q[i].toFixed(2) : ''}</span></label>`).join('');
  document.querySelectorAll('.t-q').forEach((el) => {
    el.addEventListener('pointerdown', () => { sliderDragging = true; });
    el.addEventListener('pointerup', () => { sliderDragging = false; });
    el.addEventListener('input', () => {
      el.nextElementSibling.textContent = (+el.value).toFixed(2);
      clearTimeout(pushTimer);
      pushTimer = setTimeout(() => post('/api/teach/push', { q: [...document.querySelectorAll('.t-q')].map((x) => +x.value) }).catch((e) => msg(e.message, 'err')), 40);
    });
  });
}

function renderTask(t) {
  const running = !!(t && t.running);
  $('task-phase').textContent = t ? `${t.task} · ${t.phase}${t.progress?.[1] ? ` · ${t.progress[0]}/${t.progress[1]}` : ''}` : '';
  $('btn-task-stop').disabled = !running;
  ['btn-measure', 'btn-validate', 'btn-goto', 'btn-gen'].forEach((id) => { $(id).disabled = running; });
  const log = $('task-log');
  const text = (t?.log || []).join('\n') + (t?.error ? `\n[error] ${t.error}` : '');
  if (log.textContent !== text) { log.textContent = text; log.scrollTop = log.scrollHeight; }
  if (t?.task === 'measure') {
    $('measure-progress').textContent = t.session ? `会话 ${t.session}` : '';
    const tb = $('samples-table').tBodies[0];
    const rows = ['<tr><th>姿态</th><th>逼近</th><th>指令-实测 max °</th><th>静止 s</th></tr>'].concat(
      (t.samples || []).map((s) => `<tr><td>${s.name}</td><td>${s.approach || '·'}</td><td>${fmt(s.err_max_deg, 2)}</td><td>${fmt(s.settle_s, 2)}${s.timed_out ? ' ⚠' : ''}</td></tr>`));
    tb.innerHTML = rows.join('');
    if (!running && t.result && S.lastTaskName === 'measure-running') { $('session-select').value = t.session; }
  }
  if (t?.task === 'validate') renderValidateRows(t.rows, t.summary);
  if (t?.task === 'goto' && t.result) {
    const r = t.result;
    $('goto-result').innerHTML = `最大关节误差 <b>${fmt(r.err_max_deg, 2)}°</b> · RMS ${fmt(r.err_rms_deg, 2)}° · 末端 <b>${fmt(r.ee.pos_mm, 1)} mm</b> / ${fmt(r.ee.rot_deg, 2)}° · 轨迹 ${fmt(r.traj_s, 1)}s + 静止 ${fmt(r.settle_s, 2)}s = ${fmt(r.total_s, 1)}s` +
      `<br>各关节误差 °：[${r.err_deg.join(', ')}]` + (r.corrections?.length > 1 ? `<br>修正过程：${r.corrections.map((c) => `#${c.iter} ${c.err_max_deg.toFixed(2)}°@${c.t_s.toFixed(1)}s`).join(' → ')}` : '');
  }
  S.lastTaskName = running ? `${t.task}-running` : (t?.task || null);
}

function renderSessions(list) {
  const sel = $('session-select');
  const html = (list || []).map((s) => `<option value="${s.name}">${s.name} (${s.n_samples} 样本${s.controller ? ', ' + s.controller : ''})</option>`).join('');
  if (sel.dataset.html !== html) { const v = sel.value; sel.innerHTML = html; sel.dataset.html = html; if ([...sel.options].some((o) => o.value === v)) sel.value = v; }
}

function renderPayload(st) {
  const sv = st.payload_saved;
  $('payload-saved').innerHTML = sv ? `已保存 (config/payload_${st.arm}.json)：${fmt(sv.mass_kg, 3)} kg @ [${sv.com_m.map((v) => fmt(v, 3)).join(', ')}] α=${fmt(sv.alpha, 3)} · ${sv.applied_at || ''}` : '尚无已保存的负载参数';
  $('btn-apply-saved').disabled = !sv || !st.controllable;
  $('btn-apply').disabled = !st.controllable; $('btn-clear').disabled = !st.controllable;
  if (st.last_solve && st.last_solve !== S.solve) { S.solve = st.last_solve; showSolve(st.last_solve); }
}

function showSolve(r) {
  const L = [];
  L.push(`会话 ${r.session} · ${r.n_samples} 样本 · 力矩源 ${r.source} · ${r.fit_alpha ? '拟合 α' : 'α=1'} · 条件数 ${fmt(r.regressor_cond, 1)}`);
  L.push(`质量  m  = ${fmt(r.mass_kg, 3)} ± ${fmt(r.mass_std_kg, 3)} kg`);
  L.push(`质心 com = [${r.com_m.map((v) => fmt(v, 4)).join(', ')}] m` + (r.com_std_m ? `  ±[${r.com_std_m.map((v) => fmt(v, 4)).join(', ')}]` : ''));
  if (r.fit_alpha) L.push(`臂自重比例 α = ${fmt(r.alpha, 3)} ± ${fmt(r.alpha_std, 3)}`);
  L.push(`剩余力矩 RMS：只用 URDF 前馈 ${fmt(r.residual_rms_before_nm, 2)} Nm → 加负载后 ${fmt(r.residual_rms_after_nm, 2)} Nm`);
  L.push(`折算平均下垂：${fmt(r.sag_est_before_deg, 2)}° → ${fmt(r.sag_est_after_deg, 2)}°`);
  L.push(`各关节残差 Nm(后)：[${r.residual_per_joint_after_nm.map((v) => fmt(v, 2)).join(', ')}]`);
  if (r.sim_truth) L.push(`仿真真值 m=${r.sim_truth.mass_kg} com=[${r.sim_truth.com_m.join(', ')}] 臂×${r.sim_truth.arm_mass_scale}  → 误差 Δm=${fmt(r.sim_error.mass_kg, 3)} kg，Δcom=[${r.sim_error.com_mm.map((v) => fmt(v, 1)).join(', ')}] mm`);
  const bad = r.regressor_cond > 50 || r.mass_kg <= 0;
  $('solve-result').innerHTML = `<span class="${bad ? 'bad' : 'good'}">${L.join('\n')}</span>` + (bad ? '\n⚠ 条件数过大或质量非正：姿态集不够多样，或样本太少' : '');
  $('ap-mass').value = r.mass_kg.toFixed(3); $('ap-cx').value = r.com_m[0].toFixed(4); $('ap-cy').value = r.com_m[1].toFixed(4); $('ap-cz').value = r.com_m[2].toFixed(4);
  $('ap-alpha').value = r.alpha.toFixed(3);
}

function renderValidateRows(rows, summary) {
  const names = { baseline: '基线（现状）', payload: '负载补偿', payload_correct: '补偿+修正' };
  const L = [];
  for (const [m, s] of Object.entries(summary || {})) {
    L.push(`${names[m].padEnd(8, '　')} n=${s.n}  关节误差 均值 ${fmt(s.err_max_deg_mean, 2)}° / 最差 ${fmt(s.err_max_deg_worst, 2)}°  末端 ${fmt(s.ee_mm_mean, 1)} / ${fmt(s.ee_mm_worst, 1)} mm  轨迹 ${fmt(s.traj_s_mean, 2)}s  静止 ${fmt(s.settle_s_mean, 2)}s  合计 ${fmt(s.total_s_mean, 2)}s`);
  }
  $('validate-summary').textContent = L.join('\n') || '（无结果）';
  const tb = $('validate-table').tBodies[0];
  tb.innerHTML = ['<tr><th>方案</th><th>姿态</th><th>max °</th><th>rms °</th><th>末端 mm</th><th>轨迹 s</th><th>静止 s</th><th>合计 s</th></tr>']
    .concat((rows || []).map((r) => `<tr><td>${names[r.mode]}</td><td>${r.pose}</td><td>${fmt(r.err_max_deg, 2)}</td><td>${fmt(r.err_rms_deg, 2)}</td><td>${fmt(r.ee_mm, 1)}</td><td>${fmt(r.traj_s, 2)}</td><td>${fmt(r.settle_s, 2)}</td><td>${fmt(r.total_s, 2)}</td></tr>`)).join('');
}

function renderValidations(list) {
  const sel = $('validation-select');
  const html = '<option value="">（选择查看）</option>' + (list || []).map((v) => `<option value="${v.name}">${v.name}</option>`).join('');
  if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
}

function buildGotoInputs(home) {
  const box = $('goto-inputs');
  box.innerHTML = JOINTS.map((n, i) => `<label>${n}<input type="number" step="0.01" class="g-q" data-i="${i}" value="${home ? home[i].toFixed(3) : 0}"></label>`).join('');
}
const gotoQ = () => [...document.querySelectorAll('.g-q')].map((el) => parseFloat(el.value));
const setGotoQ = (q) => document.querySelectorAll('.g-q').forEach((el, i) => { el.value = q[i].toFixed(3); });

// ---------------------------------------------------------------- 3D 视图（轨道相机，纯 canvas；世界 = torso_link：X 前 Y 左 Z 上）
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };

const V3 = {
  canvas: null, ctx: null, scene: null, yaw: -2.3, pitch: 0.4, dist: 1.6, target: [0.2, -0.05, 0.2], drag: null,
  reset() { this.yaw = -2.3; this.pitch = 0.4; this.dist = 1.6; this.target = [0.2, -0.05, 0.2]; this.draw(); },
  project(p) {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const eye = [this.target[0] + this.dist * cp * cy, this.target[1] + this.dist * cp * sy, this.target[2] + this.dist * sp];
    const f = norm3(sub3(this.target, eye)), r = norm3(cross3(f, [0, 0, 1])), u = cross3(r, f);
    const d = sub3(p, eye), x = dot3(d, r), y = dot3(d, u), z = dot3(d, f);
    if (z < 0.05) return null;
    const W = this.canvas.width, H = this.canvas.height, fl = 0.9 * H;
    return [W / 2 + (x / z) * fl, H / 2 - (y / z) * fl, z];
  },
  line(a, b, color, width = 1, dash = []) {
    const A = this.project(a), B = this.project(b); if (!A || !B) return;
    const c = this.ctx; c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash);
    c.beginPath(); c.moveTo(A[0], A[1]); c.lineTo(B[0], B[1]); c.stroke(); c.setLineDash([]);
  },
  dot(p, color, r = 3, label) {
    const P = this.project(p); if (!P) return;
    const c = this.ctx; c.fillStyle = color; c.beginPath(); c.arc(P[0], P[1], r, 0, Math.PI * 2); c.fill();
    if (label) { c.font = '11px system-ui'; c.fillText(label, P[0] + 7, P[1] - 5); }
  },
  chain(pts, color, width, dash) { for (let i = 1; i < pts.length; i++) this.line(pts[i - 1], pts[i], color, width, dash); },
  draw() {
    const c = this.ctx, cv = this.canvas, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    c.clearRect(0, 0, w, h);
    for (let g = -1; g <= 1.001; g += 0.25) { this.line([g, -1, -0.6], [g, 1, -0.6], '#1e2229'); this.line([-1, g, -0.6], [1, g, -0.6], '#1e2229'); }
    this.line([0, 0, 0], [0.3, 0, 0], '#e5484d', 2); this.line([0, 0, 0], [0, 0.3, 0], '#46a758', 2); this.line([0, 0, 0], [0, 0, 0.3], '#3e63dd', 2);
    c.fillStyle = '#9aa3ad'; c.font = '11px system-ui';
    const ax = this.project([0.33, 0, 0]); if (ax) c.fillText('X 前', ax[0], ax[1]);
    const ay = this.project([0, 0.33, 0]); if (ay) c.fillText('Y 左', ay[0], ay[1]);
    const az = this.project([0, 0, 0.33]); if (az) c.fillText('Z 上', az[0], az[1]);
    this.line([0, 0, 0.35], [0, 0, -0.5], '#3a3f48', 6); this.line([0, -0.2, 0.3], [0, 0.2, 0.3], '#3a3f48', 6);
    const sc = this.scene; if (!sc) return;
    (sc.pose_tips || []).forEach((p, i) => p && this.dot(p, 'rgba(110,231,168,0.7)', 2.5, i === 0 ? '姿态集' : undefined));
    if (sc.target_links) this.chain(sc.target_links, '#60a5fa', 2, [6, 4]);
    if (sc.cmd_links) this.chain(sc.cmd_links, '#6b7280', 2, [4, 4]);
    if (sc.links) {
      this.chain(sc.links, '#e8eaed', 4);
      sc.links.forEach((p, i) => this.dot(p, i === 0 ? '#2f6feb' : '#e8eaed', i === 0 ? 5 : 3.5));
      const wp = sc.links[sc.links.length - 1];
      if (sc.wrist_T) {   // 末端连杆坐标轴（小）
        const T = sc.wrist_T, o = [T[0][3], T[1][3], T[2][3]], L = 0.06;
        this.line(o, add3(o, [T[0][0] * L, T[1][0] * L, T[2][0] * L]), '#e5484d', 1.5);
        this.line(o, add3(o, [T[0][1] * L, T[1][1] * L, T[2][1] * L]), '#46a758', 1.5);
        this.line(o, add3(o, [T[0][2] * L, T[1][2] * L, T[2][2] * L]), '#3e63dd', 1.5);
      }
      if (sc.truth_point) { this.line(wp, sc.truth_point, '#f5b14c', 1, [2, 3]); this.dot(sc.truth_point, '#f5b14c', 4 + 3 * sc.truth_mass, `真值 ${sc.truth_mass.toFixed(2)} kg`); }
      if (sc.payload_point) {
        this.line(wp, sc.payload_point, '#e879f9', 1.5);
        this.dot(sc.payload_point, '#e879f9', 4 + 3 * sc.payload_mass, `补偿 ${sc.payload_mass.toFixed(2)} kg`);
        this.line(sc.payload_point, add3(sc.payload_point, [0, 0, -0.05 * sc.payload_mass]), '#e879f9', 2);
      }
    }
    c.font = '12px system-ui'; c.fillStyle = '#9aa3ad';
    c.fillText(`${sc.arm === 'left' ? '左臂' : '右臂'} · ${sc.kind || ''}`, 8, 16);
    if (sc.teach) { c.fillStyle = sc.teach_locked ? '#6ee7a8' : '#f5b14c'; c.fillText(sc.teach_locked ? '示教 · 已锁定（空格继续拖动）' : '示教 · 卸力中（空格锁定记录）', 8, 32); }
  },
  init() {
    this.canvas = $('scene3d'); this.ctx = this.canvas.getContext('2d'); const cv = this.canvas;
    cv.addEventListener('mousedown', (e) => { this.drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; e.preventDefault(); });
    window.addEventListener('mouseup', () => { this.drag = null; });
    window.addEventListener('mousemove', (e) => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y; this.drag.x = e.clientX; this.drag.y = e.clientY;
      if (this.drag.pan) { const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), k = this.dist * 0.0015; this.target = add3(add3(this.target, scale3([-sy, cy, 0], -dx * k)), [0, 0, dy * k]); }
      else { this.yaw -= dx * 0.008; this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.008)); }
      this.draw();
    });
    cv.addEventListener('wheel', (e) => { this.dist = Math.max(0.4, Math.min(8, this.dist * (e.deltaY > 0 ? 1.1 : 0.9))); this.draw(); e.preventDefault(); }, { passive: false });
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('dblclick', () => this.reset());
    new ResizeObserver(() => this.draw()).observe(cv);
    this.draw();
  },
};
let sceneInFlight = false;
async function pollScene() {
  if (sceneInFlight) return; sceneInFlight = true;
  try { V3.scene = await api('/api/scene'); V3.draw(); } catch (e) { /* 静默 */ } finally { sceneInFlight = false; }
}

// ---------------------------------------------------------------- 交互
$('btn-switch').onclick = () => act(async () => {
  const m = $('mode-select').value, a = $('arm-select').value;
  if (S.status?.ctrl?.engaged && !confirm('当前手臂已接管，切换会释放手臂（真机：权重渐出交还本体，请扶住）。继续？')) return;
  render(await post('/api/mode', { mode: m, arm: a }));
}, '已切换');
$('btn-engage').onclick = () => act(async () => {
  // 先做一次新鲜检测：rt/arm_sdk 上已有人发令就不接管
  const st = await api('/api/status');
  const sdk = st.arm_sdk || {};
  if (sdk.foreign && !st.ctrl?.engaged) {
    throw new Error(`rt/arm_sdk 上有其他发布者（约 ${fmt(sdk.foreign_rate_hz, 0)} Hz，权重 ${fmt(sdk.last_weight, 2)}）。先停掉对方再接管。` +
      (sdk.local_hints?.length ? ' 本机疑似：' + sdk.local_hints.map((h) => `pid ${h.pid}`).join(', ') : ''));
  }
  if (st.mode === 'real' && !confirm('⚠ 将通过 rt/arm_sdk 接管真机手臂（权重 1 s 渐入）。rt/arm_sdk 检测为空闲。确认有人守着急停？')) return;
  render(await post('/api/engage'));   // 服务端会再检测一次并拒绝
}, '已接管');
$('btn-release').onclick = () => act(async () => { if (S.status?.mode === 'real' && !confirm('释放：权重渐出交还本体控制器，请扶住手臂。继续？')) return; render(await post('/api/release')); }, '已释放');
$('btn-hold').onclick = () => act(async () => render(await post('/api/hold')), '已急停：任务停止，手臂保持');
$('btn-task-stop').onclick = () => act(() => post('/api/task/stop'), '任务停止中');
$('btn-truth').onclick = () => act(() => post('/api/sim/truth', {
  payload_mass: +$('tr-mass').value, payload_com: [+$('tr-cx').value, +$('tr-cy').value, +$('tr-cz').value],
  arm_mass_scale: +$('tr-scale').value, friction_scale: +$('tr-fric').value, foreign_publisher: $('tr-foreign').checked }), '仿真真值已写入');
$('btn-gen').onclick = () => act(async () => {
  const name = prompt('新姿态集名字（自动生成）', `auto_${S.status.arm}_${$('gen-n').value}`); if (!name) return;
  await post('/api/posesets/gen', { n: +$('gen-n').value, seed: +$('gen-seed').value, name }); pollStatus();
}, '姿态集已生成并选中');
$('btn-set-create').onclick = () => act(async () => {
  const name = $('new-set-name').value.trim(); if (!name) throw new Error('先填姿态集名字');
  await post('/api/posesets/create', { name, note: $('new-set-note').value.trim() });
  $('new-set-name').value = ''; $('new-set-note').value = ''; pollStatus();
}, '已新建并选中，现在可以示教记姿态');
$('poseset-select').onchange = () => act(async () => { await post('/api/posesets/select', { name: $('poseset-select').value }); pollStatus(); }, '已切换姿态集');
$('btn-set-delete').onclick = () => act(async () => {
  const p = S.status?.poses; if (!p) return;
  if (!confirm(`删除姿态集「${p.name}」（${p.n} 个姿态）？不可恢复。`)) return;
  await post('/api/posesets/delete', { name: p.name }); pollStatus();
}, '已删除');
$('btn-speed').onclick = () => act(() => post('/api/speed', { max_speed_rad_s: +$('speed').value }), '限速已设置');
$('pose-chips').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  const p = S.status?.poses?.poses?.[+b.dataset.i]; if (!p) return;
  act(async () => {
    if (S.status.mode === 'real' && !confirm(`真机走到 ${p.name}？`)) return;
    setGotoQ(p.q); await post('/api/goto', { q: p.q, correction_iters: 0 });
  }, `→ ${p.name}`);
};
$('pose-chips').oncontextmenu = (e) => {
  const b = e.target.closest('button'); if (!b) return; e.preventDefault();
  const i = +b.dataset.i, p = S.status?.poses?.poses?.[i]; if (!p) return;
  if (confirm(`从姿态集删除 ${p.name}？`)) act(async () => { await post('/api/posesets/pose/delete', { index: i }); pollStatus(); }, `已删除 ${p.name}`);
};
// ---- 示教 ----
$('btn-teach-still').onclick = () => act(() => post('/api/teach/still', {
  still_dq_rad_s: +$('ts-dq').value, still_drift_rad: +$('ts-drift').value,
  still_window_s: +$('ts-win').value, settle_timeout_s: +$('ts-to').value }), '静止阈值已更新');
$('btn-teach-start').onclick = () => act(async () => {
  if (S.status?.mode === 'real' && !confirm('⚠ 手臂将卸力（kp=0，只剩重力前馈）。负载未补偿前它会往下坠——请先用手扶住再确认。')) return;
  await post('/api/teach/start'); pollStatus();
}, '示教开始：手推到位后按空格');
async function teachToggle() {
  const t = S.status?.teach; if (!t?.active) return;
  await act(async () => {
    const r = await post('/api/teach/toggle');
    if (r.pose) msg(`已记录 ${r.pose.name}  q=[${r.pose.q.map((v) => v.toFixed(2)).join(', ')}]  · 按空格继续拖动`, 'ok');
    else msg('已恢复卸力，推到下一个位置后按空格', 'ok');
    pollStatus();
  });
}
$('btn-teach-toggle').onclick = teachToggle;
$('btn-teach-end').onclick = () => act(async () => { await post('/api/teach/end'); pollStatus(); }, '示教结束，手臂刚性保持');
$('btn-add-current').onclick = () => act(async () => { const r = await post('/api/posesets/add_current', {}); pollStatus(); return r; }, '已记录当前姿态');
window.addEventListener('keydown', (e) => {
  if (e.code !== 'Space') return;
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (S.status?.teach?.active) { e.preventDefault(); teachToggle(); }
});
$('btn-measure').onclick = () => act(async () => {
  if (S.status?.mode === 'real' && !confirm('真机将依次走完姿态集（每个姿态双向逼近 2 次）。确认？')) return;
  await post('/api/measure/start', { bidirectional: $('m-bidir').checked, dwell_s: +$('m-dwell').value, avg_frames: +$('m-frames').value });
}, '测量开始');
$('btn-solve').onclick = () => act(async () => {
  const r = await post('/api/solve', { session: $('session-select').value, source: $('solve-source').value, fit_alpha: $('solve-alpha').checked });
  S.solve = r; showSolve(r);
}, '求解完成');
$('btn-apply').onclick = () => act(() => post('/api/payload/apply', {
  mass_kg: +$('ap-mass').value, com_m: [+$('ap-cx').value, +$('ap-cy').value, +$('ap-cz').value], alpha: +$('ap-alpha').value,
  save: true, source_session: S.solve?.session }), '已应用到前馈并保存');
$('btn-apply-saved').onclick = () => act(() => { const s = S.status.payload_saved; return post('/api/payload/apply', { ...s, save: false }); }, '已应用已保存参数');
$('btn-clear').onclick = () => act(() => post('/api/payload/clear'), '已清除负载前馈（作者现状）');
$('btn-validate').onclick = () => act(async () => {
  const modes = [...document.querySelectorAll('.v-mode:checked')].map((el) => el.value);
  const n = +$('v-npose').value;
  const body = { modes, correction_iters: +$('v-iters').value, pose_indices: n > 0 ? [...Array(n).keys()] : null };
  if (S.status?.mode === 'real' && !confirm(`真机将按 ${modes.length} 种方案各走一遍姿态集。确认？`)) return;
  await post('/api/validate/start', body);
}, '验证开始');
$('validation-select').onchange = () => act(async () => { const n = $('validation-select').value; if (!n) return; const d = await api(`/api/validation?name=${n}`); renderValidateRows(d.rows, d.summary); });
$('btn-goto-meas').onclick = () => { const q = S.status?.ctrl?.measured_rad; if (q) setGotoQ(q); };
$('btn-goto-home').onclick = () => { const h = S.status?.poses?.home_q; if (h) setGotoQ(h); };
$('btn-goto').onclick = () => act(async () => {
  const q = gotoQ(); if (q.some((v) => !isFinite(v))) throw new Error('目标含非法数');
  if (S.status?.mode === 'real' && !confirm(`真机走到 [${q.map((v) => v.toFixed(2)).join(', ')}]？`)) return;
  $('goto-result').textContent = '…'; await post('/api/goto', { q, correction_iters: +$('g-iters').value });
}, '出发');

V3.init();
pollStatus(); setInterval(pollStatus, 400); setInterval(pollScene, 100);
