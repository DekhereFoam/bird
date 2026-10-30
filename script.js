// ทายประเภทรถด้วย YOLO (ONNX Runtime Web)
// ต้องเปิดผ่านเว็บเซิร์ฟเวอร์ (เช่น python -m http.server) ไม่ใช่ดับเบิลคลิกไฟล์

const MODEL_URL = "best.onnx";
const LABELS_URL = "labels.json";
const IGNORE_CLASSES = [];              // ใส่เลขคลาสที่ไม่ต้องการแสดง เช่น [0]
const SIZE = 640;                       // ขนาด input ของโมเดล
const NAMES_TH = { crow: "อีกา", hawk: "เหยี่ยว", pigeon: "นกพิราบ" };   // key เป็นตัวพิมพ์เล็ก
const COLORS = ["#f5c518", "#4fc3f7", "#ff7043", "#81c784", "#ba68c8"];

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const statusEl = $("status"), summaryEl = $("summary"), emptyEl = $("empty");

let session = null, labels = [], currentImg = null, loadError = "";

function setStatus(text, cls = "") { statusEl.textContent = text; statusEl.className = cls; }

// ---------- โหลดโมเดลและ labels ----------
function withTimeout(promise, ms, msg) {
  return Promise.race([promise, new Promise((_, rej) => setTimeout(() => rej(new Error(msg)), ms))]);
}

// ลองโหลด labels.json -> lables.json (สะกดผิดที่พบบ่อย) -> ใช้ค่าเริ่มต้นในโค้ด
const DEFAULT_LABELS = ["Pigeon", "crow", "hawk"];   // ลำดับตามผลเทรน train-4
async function loadLabels() {
  for (const url of [LABELS_URL, "lables.json"]) {
    try {
      const r = await fetch(url);
      if (r.ok) return await r.json();
    } catch (e) { /* ลองไฟล์ถัดไป */ }
  }
  console.warn("ไม่พบ labels.json ใช้ค่าเริ่มต้น:", DEFAULT_LABELS);
  return DEFAULT_LABELS;
}

async function init() {
  try {
    if (typeof ort === "undefined") throw new Error("โหลด onnxruntime-web จาก CDN ไม่ได้ (เช็กอินเทอร์เน็ต)");
    ort.env.wasm.numThreads = 1;   // กันปัญหา SharedArrayBuffer บน Live Server

    setStatus("กำลังโหลด labels.json...");
    labels = await loadLabels();

    setStatus("กำลังโหลด best.onnx...");
    const mres = await fetch(MODEL_URL);
    if (!mres.ok) throw new Error("ไม่พบไฟล์ best.onnx (HTTP " + mres.status + ")");
    const buf = await mres.arrayBuffer();

    setStatus("กำลังเตรียมโมเดล...");
    session = await withTimeout(
      ort.InferenceSession.create(buf, { executionProviders: ["wasm"] }),
      60000,
      "เตรียมโมเดลนานเกิน 60 วินาที"
    );
    console.log("inputs:", session.inputNames, "outputs:", session.outputNames);
    setStatus("โมเดลพร้อมแล้ว", "ok");
    if (currentImg) detect();
  } catch (e) {
    console.error(e);
    loadError = "ผิดพลาด: " + e.message;
    setStatus(loadError, "err");
  }
}
init();

// ---------- รับรูป ----------
const input = $("imageInput"), drop = $("drop");
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
input.addEventListener("change", () => input.files[0] && loadFile(input.files[0]));
$("thr").addEventListener("input", () => {
  $("thrText").textContent = $("thr").value + "%";
  if (currentImg) detect();
});

function loadFile(file) {
  const img = new Image();
  img.onload = () => {
    currentImg = img;
    $("placeholder").hidden = true;
    canvas.hidden = false;
    draw(img, []);          // แสดงรูปทันที ไม่ต้องรอโมเดล
    summaryEl.innerHTML = "";
    detect();
  };
  img.src = URL.createObjectURL(file);
}

// ---------- Preprocess (letterbox ไม่บิดสัดส่วนรูป) ----------
function preprocess(img) {
  const scale = Math.min(SIZE / img.width, SIZE / img.height);
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  const padX = (SIZE - w) / 2, padY = (SIZE - h) / 2;

  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const cx = c.getContext("2d", { willReadFrequently: true });
  cx.fillStyle = "rgb(114,114,114)";
  cx.fillRect(0, 0, SIZE, SIZE);
  cx.drawImage(img, padX, padY, w, h);

  const px = cx.getImageData(0, 0, SIZE, SIZE).data;
  const area = SIZE * SIZE, f = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    f[i] = px[i * 4] / 255;
    f[i + area] = px[i * 4 + 1] / 255;
    f[i + 2 * area] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]), scale, padX, padY };
}

// ---------- Postprocess: output [1, 4+nc, 8400] ----------
function postprocess(out, thr, pre, imgW, imgH) {
  const { data, dims } = out;
  const attrs = dims[1], n = dims[2], boxes = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = 0;
    for (let c = 4; c < attrs; c++) {
      if (IGNORE_CLASSES.includes(c - 4)) continue;
      const s = data[c * n + i];
      if (s > best) { best = s; cls = c - 4; }
    }
    if (best < thr) continue;
    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
    // แปลงกลับเป็นพิกัดบนรูปต้นฉบับ
    let x1 = (cx - w / 2 - pre.padX) / pre.scale;
    let y1 = (cy - h / 2 - pre.padY) / pre.scale;
    let x2 = (cx + w / 2 - pre.padX) / pre.scale;
    let y2 = (cy + h / 2 - pre.padY) / pre.scale;
    x1 = Math.max(0, x1); y1 = Math.max(0, y1);
    x2 = Math.min(imgW, x2); y2 = Math.min(imgH, y2);
    boxes.push({ x: x1, y: y1, width: x2 - x1, height: y2 - y1, score: best, classId: cls });
  }
  return nms(boxes, 0.45);
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width), y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter);
}

function nms(boxes, thr) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const cur = boxes.shift();
    keep.push(cur);
    boxes = boxes.filter((b) => b.classId !== cur.classId || iou(cur, b) < thr);
  }
  return keep;
}

// ---------- ตรวจจับและวาดผล ----------
async function detect() {
  try { await detectInner(); } catch (e) { console.error(e); setStatus("ผิดพลาดตอนวิเคราะห์: " + e.message, "err"); }
}

async function detectInner() {
  if (!currentImg) return;
  if (!session) { if (loadError) setStatus(loadError, "err"); return; }
  setStatus("กำลังวิเคราะห์...");
  await new Promise((r) => setTimeout(r, 20)); // ให้ UI อัปเดตก่อน

  const img = currentImg;
  const pre = preprocess(img);
  const result = await session.run({ [session.inputNames[0]]: pre.tensor });
  const thr = $("thr").value / 100;
  const out = result[session.outputNames[0]];
  const nc = out.dims[1] - 4;
  const boxes = postprocess(out, thr, pre, img.width, img.height);

  draw(img, boxes);
  showSummary(boxes);
  if (labels.length !== nc) {
    setStatus(`labels.json มี ${labels.length} ชื่อ แต่โมเดลมี ${nc} คลาส ให้แก้ labels.json ให้ครบและเรียงตามลำดับ`, "err");
  } else {
    setStatus(boxes.length ? `พบ ${boxes.length} ตัว` : "ไม่พบนกในรูป", "ok");
  }
}

function draw(img, boxes) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(img.width / 300));
  const fs = Math.max(14, Math.round(img.width / 40));
  ctx.lineWidth = lw;
  ctx.font = `600 ${fs}px "Noto Sans Thai", sans-serif`;
  ctx.textBaseline = "top";

  for (const b of boxes) {
    const color = COLORS[b.classId % COLORS.length];
    const name = labels[b.classId] || `class ${b.classId}`;
    const text = `${NAMES_TH[name.toLowerCase()] || name} ${(b.score * 100).toFixed(0)}%`;
    ctx.strokeStyle = color;
    ctx.strokeRect(b.x, b.y, b.width, b.height);

    const tw = ctx.measureText(text).width + 10, th = fs + 8;
    const ty = b.y - th < 0 ? b.y : b.y - th;
    ctx.fillStyle = color;
    ctx.fillRect(b.x, ty, tw, th);
    ctx.fillStyle = "#1c1f24";
    ctx.fillText(text, b.x + 5, ty + 4);
  }
}

function showSummary(boxes) {
  summaryEl.innerHTML = "";
  emptyEl.hidden = boxes.length > 0;
  for (const b of boxes) {
    const name = labels[b.classId] || `class ${b.classId}`;
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot" style="background:${COLORS[b.classId % COLORS.length]}"></span>
      <span class="name">${NAMES_TH[name.toLowerCase()] || name}</span>
      <span class="score">${(b.score * 100).toFixed(1)}%</span>`;
    summaryEl.appendChild(li);
  }
}