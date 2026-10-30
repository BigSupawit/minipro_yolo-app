"use client";

import { useState, useEffect, useRef } from "react";
import Dropzone from "react-dropzone";
import { UploadCloud, CheckCircle2, Play, Loader2 } from "lucide-react";
import { loadModel, runInference } from "../utils/detect";

export default function Home() {
  const [detectMode, setDetectMode] = useState<"rect" | "poly">("rect");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [session, setSession] = useState<any>(null);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const initModel = async () => {
      setIsLoadingModel(true);
      const modelUrl =
        detectMode === "rect"
          ? "/models/best_rec.onnx"
          : "/models/best-seg.onnx";
      const loadedSession = await loadModel(modelUrl);

      setSession(loadedSession);
      setIsLoadingModel(false);
    };
    initModel();
  }, [detectMode]);

  const handleImageUpload = (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    }
  };

  const handleRunDetection = async () => {
    if (!session || !imageRef.current || !canvasRef.current) return;

    setIsDetecting(true);
    try {
      await runInference(
        session,
        imageRef.current,
        canvasRef.current,
        detectMode,
      );
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการรัน AI:", error);
    } finally {
      setIsDetecting(false);
    }
  };

  

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center py-12 px-4">
      <div className="max-w-3xl w-full space-y-8">
        {/* ส่วนหัวของเว็บ */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">YOLO 👀</h1>
          <p className="text-black font-bold mt-5">
            ระบบตรวจจับสัตว์ป่า 4 ชนิด ได้แก้ กวาง กระทิง หมาใน ช้าง
          </p>

          {/* แสดงสถานะการโหลดโมเดล */}
          <div className="mt-5 inline-block">
            {isLoadingModel ? (
              <span className="text-xs text-yellow-800 px-3 py-1  font-medium flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" /> กำลังโหลดโมเดล AI (
                {detectMode === "rect" ? "Rectangle" : "Polygon"})...
              </span>
            ) : (
              <span className="text-xs  text-green-800 px-3 py-1  font-bold">
                {detectMode === "rect" ? "Rectangle" : "Polygon"} พร้อมแล้ว
              </span>
            )}
          </div>
        </div>

        {/* ปุ่มสลับโหมด (Rectangle / Polygon) */}
        <div className="flex justify-center  bg-white  shadow-sm p-1 border-black border-2">
          <button
            onClick={() => setDetectMode("rect")}
            className={`flex-1 py-3 px-1 text-sm font-bold transition-colors ${
              detectMode === "rect"
                ? "bg-black text-white"
                : "text-black hover:bg-white"
            }`}
          >
            Rectangle
          </button>
          <button
            onClick={() => setDetectMode("poly")}
            className={`flex-1 py-3 px-1  text-sm font-bold transition-colors ${
              detectMode === "poly"
                ? "bg-black text-white"
                : "text-black hover:bg-white"
            }`}
          >
            Polygon
          </button>
        </div>

        {/* พื้นที่อัปโหลดรูปภาพ */}
        <Dropzone
          onDrop={handleImageUpload}
          accept={{ "image/*": [] }}
          multiple={false}
        >
          {({ getRootProps, getInputProps }) => (
            <div
              {...getRootProps()}
              className="group border-2 border-dashed border-gray-500 p-8 text-center hover:border-black hover:bg-black transition-colors cursor-pointer bg-white"
            >
              <input {...getInputProps()} />
              <UploadCloud className="mx-auto h-12 w-12 text-black group-hover:text-white transition-colors mb-2" />
              <p className="text-black group-hover:text-white font-medium transition-colors">
                ลากรูปภาพสัตว์ป่ามาวาง หรือคลิกเพื่อเลือกไฟล์
              </p>
              <p className="text-black group-hover:text-white text-sm mt-2 transition-colors">
                รองรับไฟล์ JPG, PNG
              </p>
            </div>
          )}
        </Dropzone>

        {/* ส่วนแสดงผลรูปภาพและปุ่มสั่งรัน AI */}
        {previewUrl && (
          <div className="bg-white border-black border-2 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <h3 className="font-semibold text-black">
                  รูปภาพที่อัปโหลด
                </h3>
              </div>

              {/* ปุ่มสั่งตรวจจับภาพ */}
              <button
                onClick={handleRunDetection}
                disabled={isLoadingModel || isDetecting}
               className="bg-white hover:bg-black hover:text-white text-black px-5 py-2 border-2 border-black hover:border-black font-medium text-sm flex items-center gap-2 transition-colors disabled:opacity-50"
              >
                {isDetecting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> กำลังสแกน...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> เริ่มตรวจจับสัตว์ป่า
                  </>
                )}
              </button>
            </div>

            {/* พื้นที่ซ้อนรูปภาพและ Canvas สำหรับวาดผลลัพธ์ */}
            <div className="relative flex justify-center bg-gray-900 overflow-hidden border">
              {/* รูปภาพต้นฉบับ (ซ่อนไว้เพื่อใช้อ่านค่าพิกเซล แต่แสดงผลผ่าน Canvas แทน) */}
              <img
                ref={imageRef}
                src={previewUrl}
                alt="Preview"
                className="max-h-[500px] w-auto object-contain block"
                crossOrigin="anonymous"
              />

              {/* Canvas ที่จะใช้วาดกรอบทับ */}
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none object-contain"
              />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
