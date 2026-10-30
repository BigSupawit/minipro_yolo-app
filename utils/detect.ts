import * as ort from "onnxruntime-web";
import { ANIMAL_CLASSES } from "./classes";

const CLASS_COLORS = [
  "#22c55e", // class 0 เขียว
  "#3b82f6", // class 1 น้ำเงิน
  "#f97316", // class 2 ส้ม
  "#ef4444", // class 3 แดง
];
const getColor = (classId: number) =>
  CLASS_COLORS[classId % CLASS_COLORS.length];

const hexToRgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

export async function loadModel(modelUrl: string) {
  try {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });
    console.log(`✅ โหลดโมเดลจาก ${modelUrl} สำเร็จแล้ว!`);
    return session;
  } catch (error) {
    console.error("❌ ผิดพลาดในการโหลดโมเดล : ", error);
    return null;
  }
}

export async function preprocessImage(
  imageElement: HTMLImageElement,
): Promise<ort.Tensor> {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 640;
  const ctx = canvas.getContext("2d");

  if (!ctx) throw new Error("ไม่สามารถสร้าง Canvas Context ได้");

  ctx.drawImage(imageElement, 0, 0, 640, 640);
  const imgData = ctx.getImageData(0, 0, 640, 640);
  const pixels = imgData.data;

  const red = new Float32Array(640 * 640);
  const green = new Float32Array(640 * 640);
  const blue = new Float32Array(640 * 640);

  for (let i = 0; i < pixels.length / 4; i++) {
    red[i] = pixels[i * 4] / 255.0;
    green[i] = pixels[i * 4 + 1] / 255.0;
    blue[i] = pixels[i * 4 + 2] / 255.0;
  }

  const inputData = new Float32Array(3 * 640 * 640);
  inputData.set(red, 0);
  inputData.set(green, 640 * 640);
  inputData.set(blue, 2 * 640 * 640);

  return new ort.Tensor("float32", inputData, [1, 3, 640, 640]);
}

export async function runInference(
  session: ort.InferenceSession,
  imageElement: HTMLImageElement,
  canvasElement: HTMLCanvasElement,
  detectMode: "rect" | "poly",
) {
  if (!session) return;

  const tensor = await preprocessImage(imageElement);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;

  const results = await session.run(feeds);

  console.log("outputNames:", session.outputNames);
  for (const name of session.outputNames) {
    console.log(name, results[name].dims);
  }

  const ctx = canvasElement.getContext("2d");
  if (!ctx) return;

  canvasElement.width = imageElement.clientWidth;
  canvasElement.height = imageElement.clientHeight;

  ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  ctx.drawImage(imageElement, 0, 0, canvasElement.width, canvasElement.height);

  const scaleX = canvasElement.width / 640;
  const scaleY = canvasElement.height / 640;

  if (detectMode === "rect") {
    const output = results[session.outputNames[0]];
    const outputData = output.data as Float32Array;
    const numCols = output.dims[1]; // 4 + nc
    const numRows = output.dims[2]; // 8400

    type Box = {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      score: number;
      classId: number;
    };
    const boxes: Box[] = [];

    for (let i = 0; i < numRows; i++) {
      let maxScore = 0,
        classId = 0;
      for (let c = 4; c < numCols; c++) {
        const s = outputData[c * numRows + i];
        if (s > maxScore) {
          maxScore = s;
          classId = c - 4;
        }
      }
      if (maxScore < 0.5) continue;

      const cx = outputData[0 * numRows + i];
      const cy = outputData[1 * numRows + i];
      const w = outputData[2 * numRows + i];
      const h = outputData[3 * numRows + i];
      boxes.push({
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        score: maxScore,
        classId,
      });
    }

    // ---- NMS ----
    const iou = (a: Box, b: Box) => {
      const ix1 = Math.max(a.x1, b.x1),
        iy1 = Math.max(a.y1, b.y1);
      const ix2 = Math.min(a.x2, b.x2),
        iy2 = Math.min(a.y2, b.y2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      return inter / (areaA + areaB - inter + 1e-6);
    };
    boxes.sort((a, b) => b.score - a.score);
    const kept: Box[] = [];
    for (const b of boxes) {
      if (kept.every((k) => k.classId !== b.classId || iou(k, b) < 0.45))
        kept.push(b);
    }

    // ---- วาด ----
    ctx.font = "12px sans-serif";

    for (const b of kept) {
      const color = getColor(b.classId);
      const x = b.x1 * scaleX,
        y = b.y1 * scaleY;
      const w = (b.x2 - b.x1) * scaleX,
        h = (b.y2 - b.y1) * scaleY;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      const label = `${ANIMAL_CLASSES[b.classId] ?? `Class ${b.classId}`} (${Math.round(b.score * 100)}%)`;
      const tw = ctx.measureText(label).width + 8;
      const ly = y > 20 ? y - 20 : 0;

      ctx.fillStyle = color;
      ctx.fillRect(x, ly, tw, 20);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x + 4, ly + 14);
    }
  } else {
    const output = results[session.outputNames[0]];
    const outputData = output.data as Float32Array;

    const NUM_MASKS = 32;
    const numChannels = output.dims[1]; // 4 + nc + 32
    const numRows = output.dims[2]; // 8400
    const numClasses = numChannels - 4 - NUM_MASKS;
    const maskStart = 4 + numClasses;

    const protoOutput = results[session.outputNames[1]];
    if (!protoOutput) {
      throw new Error(
        `โมเดลนี้ไม่ใช่ segmentation (มี output ${session.outputNames.length} ตัว: ${session.outputNames.join(", ")}) กรุณาใช้ไฟล์ -seg.onnx`,
      );
    }
    const protoData = protoOutput.data as Float32Array;
    const [, , protoH, protoW] = protoOutput.dims as number[]; // 1,32,160,160

    // modelSize = ขนาด input ที่ป้อนเข้าโมเดล (เช่น 640)
    const modelSize = 640;

    type Det = {
      x1: number;
      y1: number;
      x2: number;
      y2: number; // พิกัดในสเกลโมเดล (640)
      score: number;
      classId: number;
      coeffs: Float32Array;
    };
    let dets: Det[] = [];

    for (let i = 0; i < numRows; i++) {
      let maxScore = 0,
        classId = 0;
      for (let c = 4; c < maskStart; c++) {
        const s = outputData[c * numRows + i];
        if (s > maxScore) {
          maxScore = s;
          classId = c - 4;
        }
      }
      if (maxScore < 0.5) continue;

      const cx = outputData[0 * numRows + i];
      const cy = outputData[1 * numRows + i];
      const w = outputData[2 * numRows + i];
      const h = outputData[3 * numRows + i];

      const coeffs = new Float32Array(NUM_MASKS);
      for (let m = 0; m < NUM_MASKS; m++) {
        coeffs[m] = outputData[(maskStart + m) * numRows + i];
      }

      dets.push({
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        score: maxScore,
        classId,
        coeffs,
      });
    }

    // ---- NMS ----
    const iou = (a: Det, b: Det) => {
      const ix1 = Math.max(a.x1, b.x1),
        iy1 = Math.max(a.y1, b.y1);
      const ix2 = Math.min(a.x2, b.x2),
        iy2 = Math.min(a.y2, b.y2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      return inter / (areaA + areaB - inter + 1e-6);
    };
    dets.sort((a, b) => b.score - a.score);
    const kept: Det[] = [];
    for (const d of dets) {
      if (kept.every((k) => k.classId !== d.classId || iou(k, d) < 0.45))
        kept.push(d);
    }

    // ---- สร้าง mask + วาด ----
    for (const d of kept) {

      const color = getColor(d.classId);
      const [r, g, b] = hexToRgb(color);
      // 1) mask = sigmoid(coeffs · proto)
      const mask = new Float32Array(protoH * protoW);
      for (let p = 0; p < protoH * protoW; p++) {
        let sum = 0;
        for (let m = 0; m < NUM_MASKS; m++) {
          sum += d.coeffs[m] * protoData[m * protoH * protoW + p];
        }
        mask[p] = 1 / (1 + Math.exp(-sum));
      }

      // 2) แปลง bbox ไปสเกลของ proto แล้วจำกัด mask ให้อยู่ในกล่อง
      const sx = protoW / modelSize,
        sy = protoH / modelSize;
      const bx1 = Math.max(0, Math.floor(d.x1 * sx)),
        by1 = Math.max(0, Math.floor(d.y1 * sy));
      const bx2 = Math.min(protoW, Math.ceil(d.x2 * sx)),
        by2 = Math.min(protoH, Math.ceil(d.y2 * sy));

      // 3) วาด mask เป็นสีทับ (ใช้ canvas ชั่วคราวขนาดเท่า proto แล้ว scale ขึ้น)
      const mc = document.createElement("canvas");
      mc.width = protoW;
      mc.height = protoH;
      const mctx = mc.getContext("2d")!;
      const img = mctx.createImageData(protoW, protoH);
      for (let y = by1; y < by2; y++) {
        for (let x = bx1; x < bx2; x++) {
          if (mask[y * protoW + x] > 0.5) {
            const k = (y * protoW + x) * 4;
            img.data[k] = r;
            img.data[k + 1] = g;
            img.data[k + 2] = b;
            img.data[k + 3] = 110;
          }
        }
      }
      mctx.putImageData(img, 0, 0);
      ctx.drawImage(mc, 0, 0, ctx.canvas.width, ctx.canvas.height);

      // 4) ป้ายชื่อ (แปลงพิกัดโมเดล → canvas)
      const X = d.x1 * scaleX,
        Y = d.y1 * scaleY;
      const label = `${ANIMAL_CLASSES[d.classId] ?? d.classId} ${Math.round(d.score * 100)}%`;
      ctx.fillStyle = color;
      ctx.fillRect(
        X,
        Y > 22 ? Y - 22 : 0,
        ctx.measureText(label).width + 12,
        20,
      );
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(label, X + 6, Y > 22 ? Y - 7 : 14);
    }
  }
  console.log(`🎯 ประมวลผลสำเร็จในโหมด: ${detectMode}`);
}

