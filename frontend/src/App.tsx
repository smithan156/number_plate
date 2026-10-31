import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type ComponentProps, type ReactNode } from "react";
import {
  Activity, AlertCircle, AlertTriangle, Bike, Bus, Camera,
  Car, CarFront, CheckCircle2, ChevronLeft, ChevronRight,
  Download, FileImage, FileVideo, LayoutDashboard, Menu,
  Radio, RotateCcw, Scan, Search, Settings, ShieldCheck, SlidersHorizontal,
  Sparkles, Tractor, Truck, Upload, Video, X, ZoomIn,
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const WS = API.replace(/^http/, "ws");

type Detection = {
  id: number;
  vehicle_type: string;
  vehicle_confidence: number;
  plate_text: string | null;
  plate_confidence: number | null;
  ocr_confidence: number | null;
  status: "recognized" | "low_confidence" | "plate_not_detected" | string;
  source: string;
  image_url: string;
  annotated_url: string | null;
  plate_image_url: string | null;
  vehicle_box: number[];
  plate_box: number[] | null;
  created_at: string;
};

type Stats = {
  total_vehicles: number;
  plates_recognized: number;
  cars_detected: number;
  bikes_detected: number;
  trucks_detected: number;
  recognition_rate: number;
  by_vehicle_type: Record<string, number>;
  daily_counts: Array<{ date: string; count: number }>;
  hourly_counts: Array<{ hour: number; count: number }>;
};

type Health = { status: string; models: Record<string, boolean> };
type TestSample = { name: string; url: string; size: number };

type NavItem = { label: string; icon: typeof LayoutDashboard; section?: string };
const navItems: NavItem[] = [
  { label: "AI Detection", icon: Scan, section: "CORE SYSTEM" },
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Live detection", icon: Radio },
  { label: "Video analysis", icon: FileVideo },
  { label: "Vehicle records", icon: CarFront, section: "INTELLIGENCE" },
  { label: "Plate search", icon: Search },
  { label: "Analytics", icon: Activity },
  { label: "Reports", icon: Download, section: "MANAGEMENT" },
  { label: "Settings", icon: Settings },
];

const chartColors = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#c084fc", "#f87171", "#94a3b8"];
const emptyStats: Stats = {
  total_vehicles: 0, plates_recognized: 0, cars_detected: 0, bikes_detected: 0,
  trucks_detected: 0, recognition_rate: 0, by_vehicle_type: {}, daily_counts: [], hourly_counts: [],
};

function mediaUrl(path: string | null) {
  return path ? `${API}${path}` : "";
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString([], { day: "2-digit", month: "short", year: "numeric" });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function App() {
  const [page, setPage] = useState("AI Detection");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [stats, setStats] = useState<Stats>(emptyStats);
  const [health, setHealth] = useState<Health | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [vehicleFilter, setVehicleFilter] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [pageIndex, setPageIndex] = useState(1);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<Detection | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [nextStats, nextHealth] = await Promise.all([
        request<Stats>("/api/stats"),
        request<Health>("/api/health"),
      ]);
      setStats(nextStats);
      setHealth(nextHealth);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not connect to the ANPR API.");
    }
    setLoading(false);
  }, []);

  const loadDetections = useCallback(async () => {
    const params = new URLSearchParams({ page: String(pageIndex), page_size: "10", sort_by: "created_at", order: "desc" });
    if (query.trim()) params.set("q", query.trim());
    if (vehicleFilter) params.set("vehicle_type", vehicleFilter);
    if (fromDate) params.set("date_from", fromDate);
    if (toDate) params.set("date_to", toDate);
    try {
      const result = await request<{ items: Detection[]; total: number }>(`/api/detections?${params}`);
      setDetections(result.items);
      setTotal(result.total);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load vehicle records.");
    }
  }, [fromDate, pageIndex, query, toDate, vehicleFilter]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { void loadDetections(); }, [loadDetections]);
  useEffect(() => {
    const timer = window.setInterval(() => { void refresh(); void loadDetections(); }, 15000);
    return () => window.clearInterval(timer);
  }, [loadDetections, refresh]);

  const changePage = (next: string) => {
    setPage(next);
    setMobileOpen(false);
    setPageIndex(1);
  };

  const exportCsv = () => {
    const params = new URLSearchParams();
    if (query.trim()) params.set("q", query.trim());
    if (vehicleFilter) params.set("vehicle_type", vehicleFilter);
    if (fromDate) params.set("date_from", fromDate);
    if (toDate) params.set("date_to", toDate);
    window.location.href = `${API}/api/detections/export.csv?${params}`;
  };

  const tableProps = {
    detections, total, pageIndex, onPage: setPageIndex, onSelect: setSelected,
    onExport: exportCsv, query, setQuery, vehicleFilter, setVehicleFilter,
    fromDate, setFromDate, toDate, setToDate,
  };

  return (
    <div className="min-h-screen bg-[#060b14] text-slate-100 antialiased">
      {mobileOpen && (
        <button
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-30 bg-black/70 backdrop-blur-sm lg:hidden"
        />
      )}

      <Sidebar active={page} onChange={changePage} mobileOpen={mobileOpen} health={health} />

      <main className="min-h-screen lg:ml-[264px]">
        <Topbar
          page={page}
          onMenu={() => setMobileOpen(true)}
          query={query}
          setQuery={(value) => { setQuery(value); setPageIndex(1); }}
          health={health}
        />

        <div className="mx-auto max-w-[1600px] px-4 pb-12 pt-6 sm:px-7 lg:px-9">
          {error && (
            <div className="mb-6 flex items-center justify-between gap-3 rounded-xl border border-rose-500/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              <div className="flex items-center gap-2.5">
                <AlertCircle size={18} className="shrink-0 text-rose-400" />
                <span>{error}. Ensure the FastAPI backend server is running on port 8000.</span>
              </div>
              <button aria-label="Dismiss error" onClick={() => setError("")} className="text-rose-400 hover:text-white">
                <X size={16} />
              </button>
            </div>
          )}

          {page === "AI Detection" && (
            <PlateDetectionStudio onDetectionSaved={() => { void refresh(); void loadDetections(); }} />
          )}
          {page === "Dashboard" && (
            <Dashboard stats={stats} health={health} loading={loading} onNavigate={changePage} tableProps={tableProps} />
          )}
          {page === "Live detection" && <LiveDetection health={health} />}
          {page === "Video analysis" && (
            <UploadPage kind="video" onComplete={() => { void refresh(); void loadDetections(); }} />
          )}
          {(page === "Vehicle records" || page === "Plate search") && (
            <RecordsPage title={page} {...tableProps} />
          )}
          {page === "Analytics" && <Analytics stats={stats} />}
          {page === "Reports" && <Reports stats={stats} detections={detections} onExport={exportCsv} />}
          {page === "Settings" && <SettingsPage health={health} />}

          <footer className="mt-14 flex flex-col justify-between gap-3 border-t border-slate-800/80 pt-6 text-[11px] text-slate-500 sm:flex-row">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-400">AI VEHICLE NUMBER PLATE DETECTION</span>
              <span>·</span>
              <span>Ultralytics YOLO11 &amp; EasyOCR Engine</span>
            </div>
            <span>High-precision local inference · Private edge processing</span>
          </footer>
        </div>
      </main>

      {selected && <DetectionDetail detection={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

// ==========================================
// PRIMARY AI DETECTION STUDIO COMPONENT
// ==========================================
function PlateDetectionStudio({ onDetectionSaved }: { onDetectionSaved: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileMeta, setFileMeta] = useState<{ name: string; size: string; type: string } | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionStep, setDetectionStep] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [response, setResponse] = useState<{
    detections: Detection[];
    image_url: string | null;
    annotated_url: string | null;
    message: string | null;
  } | null>(null);
  const [testSamples, setTestSamples] = useState<TestSample[]>([]);
  const [showFullAnnotated, setShowFullAnnotated] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch test samples
  useEffect(() => {
    fetch(`${API}/api/test-samples`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: TestSample[]) => setTestSamples(data))
      .catch(() => setTestSamples([]));
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a supported image file (JPG, JPEG, or PNG).");
      return;
    }
    setError("");
    setFile(selectedFile);
    setResponse(null);

    const sizeKb = (selectedFile.size / 1024).toFixed(1);
    setFileMeta({
      name: selectedFile.name,
      size: `${sizeKb} KB`,
      type: selectedFile.type,
    });

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const chosen = event.target.files?.[0];
    if (chosen) handleFileSelect(chosen);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFileSelect(dropped);
  };

  const handleLoadSample = async (sample: TestSample) => {
    try {
      setError("");
      setIsDetecting(true);
      setDetectionStep("Loading sample image...");
      const res = await fetch(`${API}${sample.url}`);
      const blob = await res.blob();
      const sampleFile = new File([blob], sample.name, { type: "image/jpeg" });
      handleFileSelect(sampleFile);
      setIsDetecting(false);
    } catch {
      setIsDetecting(false);
      setError("Failed to load sample image.");
    }
  };

  const handleDetect = async () => {
    if (!file) {
      setError("Please choose or upload a vehicle image first.");
      return;
    }

    setIsDetecting(true);
    setError("");
    setResponse(null);

    // Simulate animated scanning stages for awesome UX
    setDetectionStep("1. Running YOLO11 Vehicle Detection...");
    const timer1 = setTimeout(() => {
      setDetectionStep("2. Locating Number Plate Bounding Box...");
    }, 400);
    const timer2 = setTimeout(() => {
      setDetectionStep("3. Extracting Characters via OCR Engine...");
    }, 850);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const result = await request<{
        detections: Detection[];
        image_url: string | null;
        annotated_url: string | null;
        message: string | null;
      }>("/api/detections/image", {
        method: "POST",
        body: formData,
      });

      setResponse(result);
      onDetectionSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Vehicle & plate detection failed.");
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      setIsDetecting(false);
      setDetectionStep("");
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setFileMeta(null);
    setResponse(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDownload = async () => {
    const downloadTarget = response?.annotated_url || response?.image_url;
    if (!downloadTarget) return;
    try {
      const fullUrl = mediaUrl(downloadTarget);
      const res = await fetch(fullUrl);
      const blob = await res.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = `plate-detected-${Date.now()}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(blobUrl);
    } catch {
      window.open(mediaUrl(downloadTarget), "_blank");
    }
  };

  const [selectedIdx, setSelectedIdx] = useState(0);

  // Auto-select the vehicle that has a plate whenever a new detection response arrives
  useEffect(() => {
    if (response?.detections?.length) {
      const idxWithPlate = response.detections.findIndex((d) => Boolean(d.plate_text || d.plate_box));
      setSelectedIdx(idxWithPlate !== -1 ? idxWithPlate : 0);
    } else {
      setSelectedIdx(0);
    }
  }, [response]);

  const activeDetection = response?.detections?.[selectedIdx] || null;
  const hasPlateFound = Boolean(activeDetection?.plate_box || activeDetection?.plate_image_url);
  const isPlateRecognized = Boolean(activeDetection?.plate_text) && (activeDetection?.status === "recognized" || (activeDetection?.ocr_confidence || 0) >= 0.20);
  const isManualReview = hasPlateFound && !isPlateRecognized;
  const isNoPlate = Boolean(response && (!activeDetection || activeDetection.status === "plate_not_detected" || !hasPlateFound));

  return (
    <div className="space-y-6">
      {/* HEADER SECTION */}
      <div className="flex flex-col justify-between gap-4 border-b border-cyan-500/15 pb-6 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.25em] text-cyan-400">
            <span className="flex h-2 w-2 rounded-full bg-cyan-400 soft-pulse" />
            YOLO11 &amp; OCR Engine Active
          </div>
          <h1 className="mt-1 font-display text-2xl font-extrabold tracking-tight text-white sm:text-3xl lg:text-4xl">
            AI VEHICLE NUMBER PLATE DETECTION
          </h1>
          <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400 sm:text-sm">
            Deep-learning automated vehicle category detection, custom license plate localization, and character recognition using PyTorch and OpenCV.
          </p>
        </div>

        {response && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-xs font-semibold text-slate-200 transition hover:border-cyan-500/40 hover:bg-slate-700 hover:text-white"
            >
              <RotateCcw size={14} className="text-cyan-400" />
              Reset &amp; Upload Another
            </button>
            <button
              onClick={handleDownload}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-4 py-2.5 text-xs font-bold text-white shadow-lg shadow-cyan-500/20 transition hover:scale-[1.02] hover:shadow-cyan-500/40"
            >
              <Download size={14} />
              Download Processed
            </button>
          </div>
        )}
      </div>

      {/* SAMPLE QUICK-SELECTOR BAR */}
      {testSamples.length > 0 && !response && (
        <div className="flex flex-wrap items-center gap-2.5 rounded-xl border border-cyan-500/20 bg-slate-900/60 p-3.5 backdrop-blur-md">
          <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
            <Sparkles size={14} className="text-cyan-400" />
            Quick Test Samples:
          </span>
          <div className="flex flex-wrap gap-2">
            {testSamples.map((sample, idx) => (
              <button
                key={sample.name}
                onClick={() => handleLoadSample(sample)}
                disabled={isDetecting}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-800/60 px-3 py-1.5 text-[11px] font-medium text-slate-300 transition hover:border-cyan-400/50 hover:bg-cyan-500/10 hover:text-cyan-200 disabled:opacity-50"
              >
                <CarFront size={13} className="text-cyan-400" />
                Sample {idx + 1}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* MAIN TWO-COLUMN STUDIO */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* LEFT COLUMN: UPLOAD & PREVIEW AREA */}
        <div className="space-y-5 lg:col-span-5">
          <div className="glass-card overflow-hidden rounded-2xl p-5 sm:p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2 font-display text-sm font-bold text-white">
                <Upload size={16} className="text-cyan-400" />
                1. Upload Vehicle Image
              </div>
              <span className="rounded-md border border-cyan-400/20 bg-cyan-400/10 px-2 py-0.5 text-[10px] font-semibold text-cyan-300">
                JPG, JPEG, PNG
              </span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              className="hidden"
              onChange={handleInputChange}
            />

            {!previewUrl ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                className="group relative flex min-h-[290px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700/80 bg-slate-900/40 p-6 text-center transition hover:border-cyan-400/60 hover:bg-cyan-500/[0.03]"
              >
                <div className="mb-4 grid h-16 w-16 place-items-center rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 text-cyan-400 transition group-hover:scale-110 group-hover:border-cyan-400/50 group-hover:shadow-lg group-hover:shadow-cyan-500/20">
                  <FileImage size={28} />
                </div>
                <div className="font-display text-base font-bold text-slate-100">
                  Drag and drop vehicle image here
                </div>
                <p className="mt-1.5 max-w-[280px] text-xs text-slate-500">
                  Select a clear image of a car, bike, truck, bus, auto, van, or tractor.
                </p>
                <span className="mt-5 inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-500 px-4 py-2 text-xs font-bold text-white shadow-md shadow-cyan-500/20 transition group-hover:scale-105">
                  <Upload size={14} />
                  Browse Files
                </span>
              </div>
            ) : (
              <div className="space-y-4">
                {/* IMAGE PREVIEW AREA */}
                <div className="relative overflow-hidden rounded-xl border border-cyan-500/30 bg-black/60">
                  <img
                    src={previewUrl}
                    alt="Vehicle preview"
                    className="max-h-[360px] w-full object-contain"
                  />

                  {/* SCANNING RADAR OVERLAY DURING INFERENCE */}
                  {isDetecting && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/75 backdrop-blur-sm">
                      <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent scanner-bar shadow-[0_0_15px_#22d3ee]" />
                      <div className="relative grid h-16 w-16 place-items-center rounded-full border border-cyan-400/30 bg-cyan-500/10">
                        <Scan size={28} className="text-cyan-300 animate-pulse" />
                        <div className="absolute inset-0 rounded-full border border-cyan-400/50 radar-sweep border-t-transparent" />
                      </div>
                      <div className="mt-4 font-display text-sm font-bold text-white">
                        AI Model Processing
                      </div>
                      <div className="mt-1 font-mono text-xs text-cyan-300">
                        {detectionStep || "Analyzing frame..."}
                      </div>
                    </div>
                  )}

                  <div className="absolute bottom-2 right-2 flex items-center gap-1.5 rounded-md bg-black/80 px-2 py-1 text-[10px] font-mono text-slate-300 backdrop-blur-sm">
                    <span>{fileMeta?.name}</span>
                    <span>·</span>
                    <span className="text-cyan-400">{fileMeta?.size}</span>
                  </div>
                </div>

                {/* PREVIEW CONTROLS */}
                <div className="flex items-center justify-between text-xs">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isDetecting}
                    className="font-medium text-slate-400 underline-offset-4 hover:text-cyan-300 hover:underline disabled:opacity-50"
                  >
                    Change Image
                  </button>
                  <button
                    onClick={handleReset}
                    disabled={isDetecting}
                    className="font-medium text-rose-400 underline-offset-4 hover:text-rose-300 hover:underline disabled:opacity-50"
                  >
                    Remove
                  </button>
                </div>
              </div>
            )}

            {/* ERROR DISPLAY */}
            {error && (
              <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-200">
                <AlertCircle size={16} className="mt-0.5 shrink-0 text-rose-400" />
                <span>{error}</span>
              </div>
            )}

            {/* LARGE DETECT NUMBER PLATE BUTTON */}
            <div className="mt-5">
              <button
                onClick={handleDetect}
                disabled={!file || isDetecting}
                className="group relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-xl bg-gradient-to-r from-blue-600 via-cyan-500 to-blue-500 py-3.5 text-center font-display text-sm font-extrabold uppercase tracking-wider text-white shadow-xl shadow-cyan-500/25 transition-all duration-300 hover:scale-[1.01] hover:shadow-cyan-500/40 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isDetecting ? (
                  <>
                    <Activity size={18} className="animate-spin text-white" />
                    <span>Analyzing Image...</span>
                  </>
                ) : (
                  <>
                    <Scan size={18} className="transition-transform group-hover:scale-110" />
                    <span>Detect Number Plate</span>
                    <Sparkles size={16} className="text-cyan-200" />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* DETECTION PIPELINE SPECIFICATION */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-xs text-slate-400">
            <div className="flex items-center gap-2 font-semibold text-slate-300">
              <ShieldCheck size={15} className="text-cyan-400" />
              Verified Local Detection Workflow
            </div>
            <div className="mt-2.5 flex items-center justify-between text-[11px] text-slate-500">
              <span className="flex items-center gap-1"><span className="text-cyan-400">1.</span> Vehicle YOLO</span>
              <span>→</span>
              <span className="flex items-center gap-1"><span className="text-cyan-400">2.</span> Plate YOLO</span>
              <span>→</span>
              <span className="flex items-center gap-1"><span className="text-cyan-400">3.</span> Crop &amp; Preprocess</span>
              <span>→</span>
              <span className="flex items-center gap-1"><span className="text-cyan-400">4.</span> EasyOCR</span>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: DETECTION RESULTS DISPLAY */}
        <div className="space-y-5 lg:col-span-7">
          {!response ? (
            <div className="glass-card flex min-h-[460px] flex-col items-center justify-center rounded-2xl p-8 text-center">
              <div className="mb-4 grid h-16 w-16 place-items-center rounded-2xl border border-slate-700/80 bg-slate-800/40 text-slate-500">
                <Scan size={30} />
              </div>
              <h2 className="font-display text-lg font-bold text-slate-300">
                Awaiting Vehicle Image Input
              </h2>
              <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-slate-500">
                Upload or drop a vehicle photo on the left, then click{" "}
                <span className="font-semibold text-cyan-400">"Detect Number Plate"</span>.
                Actual predictions, cropped plate, vehicle class, and confidence scores will appear here.
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              {/* MULTI-VEHICLE SELECTOR TABS (IF MULTIPLE VEHICLES DETECTED) */}
              {response.detections.length > 1 && (
                <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                  <span className="text-xs font-semibold text-slate-400">
                    Detected Vehicles ({response.detections.length}):
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {response.detections.map((det, idx) => (
                      <button
                        key={det.id || idx}
                        onClick={() => setSelectedIdx(idx)}
                        className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                          selectedIdx === idx
                            ? "border border-cyan-400/50 bg-cyan-500/20 text-cyan-200 shadow-sm"
                            : "border border-slate-700/80 bg-slate-800/60 text-slate-400 hover:text-white"
                        }`}
                      >
                        <VehicleIcon type={det.vehicle_type} />
                        <span>{det.vehicle_type}</span>
                        {det.plate_text ? (
                          <span className="font-mono font-bold text-cyan-300">[{det.plate_text}]</span>
                        ) : (
                          <span className="text-slate-500">[No Plate]</span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* HIGHLIGHTED NUMBER PLATE CARD */}
              <div className="glass-card overflow-hidden rounded-2xl p-6 shadow-2xl">
                <div className="flex items-center justify-between">
                  <div className="text-[11px] font-bold tracking-widest text-slate-400">
                    RECOGNIZED REGISTRATION NUMBER
                  </div>
                  {isPlateRecognized ? (
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-bold text-emerald-300">
                      <CheckCircle2 size={13} />
                      OCR Verified
                    </span>
                  ) : activeDetection?.plate_text ? (
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-bold text-cyan-300">
                      <Scan size={13} />
                      Recognized Plate
                    </span>
                  ) : isManualReview ? (
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1 text-xs font-bold text-amber-300">
                      <AlertTriangle size={13} />
                      Manual Verification Required
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-400/30 bg-rose-400/10 px-3 py-1 text-xs font-bold text-rose-300">
                      <AlertCircle size={13} />
                      Number Plate Not Found
                    </span>
                  )}
                </div>

                {/* REALISTIC EMBOSSED LICENSE PLATE DISPLAY */}
                <div className="mt-4 flex items-center justify-center py-2">
                  {isNoPlate && !activeDetection?.plate_text ? (
                    <div className="w-full rounded-xl border border-rose-500/25 bg-rose-500/10 p-5 text-center">
                      <div className="font-display text-lg font-bold text-rose-300">
                        Number Plate Not Found
                      </div>
                      <div className="mt-1 text-xs text-rose-200/70">
                        No license plate was detected by YOLO on this vehicle frame.
                      </div>
                    </div>
                  ) : (
                    <div className="relative flex w-full max-w-lg items-center overflow-hidden rounded-xl border-4 border-slate-900 bg-gradient-to-b from-white to-slate-200 px-4 py-3 shadow-2xl sm:px-6 sm:py-4">
                      {/* Left IND badge */}
                      <div className="mr-4 flex flex-col items-center justify-center border-r-2 border-slate-400/40 pr-3 sm:mr-6 sm:pr-4">
                        <div className="h-3 w-4 rounded-xs bg-blue-700 flex items-center justify-center">
                          <span className="text-[6px] font-black text-white">IND</span>
                        </div>
                        <span className="mt-1 text-[9px] font-black tracking-widest text-blue-900">IND</span>
                      </div>

                      {/* Plate text in large bold font */}
                      <div className="min-w-0 flex-1 text-center font-mono-plate font-black tracking-[0.18em] text-slate-950">
                        {activeDetection?.plate_text ? (
                          <span className="text-2xl sm:text-3xl lg:text-4xl text-slate-950">
                            {activeDetection.plate_text}
                          </span>
                        ) : isManualReview ? (
                          <div className="flex flex-col items-center">
                            <span className="text-xl sm:text-2xl font-bold text-amber-900">
                              MANUAL VERIFICATION REQUIRED
                            </span>
                            <span className="mt-1 text-[10px] font-mono font-semibold text-amber-700">
                              Plate located, but character contrast is low — Inspect crop below
                            </span>
                          </div>
                        ) : (
                          <span className="text-xl font-bold text-slate-600">UNRECOGNIZED</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* THREE METRICS CARDS: VEHICLE, PLATE, AND OCR CONFIDENCE */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                {/* Vehicle Category Card */}
                <div className="glass-card rounded-xl p-4">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Vehicle Category</span>
                    <VehicleIcon type={activeDetection?.vehicle_type || "Vehicle"} />
                  </div>
                  <div className="mt-2 font-display text-lg font-bold text-slate-100">
                    {activeDetection?.vehicle_type || "None Detected"}
                  </div>
                  <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                    <span>YOLO Confidence:</span>
                    <span className="font-semibold text-cyan-300">
                      {activeDetection ? `${Math.round(activeDetection.vehicle_confidence * 100)}%` : "0%"}
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-cyan-400"
                      style={{ width: `${activeDetection ? activeDetection.vehicle_confidence * 100 : 0}%` }}
                    />
                  </div>
                </div>

                {/* Plate Detection Card */}
                <div className="glass-card rounded-xl p-4">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Plate Detection</span>
                    <Scan size={16} className="text-emerald-400" />
                  </div>
                  <div className="mt-2 font-display text-lg font-bold text-slate-100">
                    {hasPlateFound ? "Plate Located" : "Not Found"}
                  </div>
                  <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                    <span>Detector Conf:</span>
                    <span className="font-semibold text-emerald-300">
                      {activeDetection?.plate_confidence
                        ? `${Math.round(activeDetection.plate_confidence * 100)}%`
                        : hasPlateFound
                        ? "Located"
                        : "0%"}
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-emerald-400"
                      style={{
                        width: `${
                          activeDetection?.plate_confidence
                            ? activeDetection.plate_confidence * 100
                            : hasPlateFound
                            ? 85
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>

                {/* OCR Confidence Card */}
                <div className="glass-card rounded-xl p-4">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>OCR Confidence</span>
                    <Activity size={16} className="text-blue-400" />
                  </div>
                  <div className="mt-2 font-display text-lg font-bold text-slate-100">
                    {activeDetection?.ocr_confidence !== null && activeDetection?.ocr_confidence !== undefined
                      ? `${Math.round(activeDetection.ocr_confidence * 100)}%`
                      : activeDetection?.plate_text
                      ? "Detected"
                      : "—"}
                  </div>
                  <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                    <span>Read Accuracy:</span>
                    <span className="font-semibold text-blue-300">
                      {activeDetection?.ocr_confidence ? `${Math.round(activeDetection.ocr_confidence * 100)}%` : "N/A"}
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-blue-500"
                      style={{ width: `${(activeDetection?.ocr_confidence || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* CROPPED NUMBER PLATE & PROCESSED IMAGE DISPLAY */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-12">
                {/* Cropped Plate Preview */}
                <div className="glass-card flex flex-col rounded-xl p-4 sm:col-span-4">
                  <div className="mb-2 text-[11px] font-bold tracking-wider text-slate-400">
                    CROPPED NUMBER PLATE
                  </div>
                  <div className="flex min-h-[120px] flex-1 items-center justify-center overflow-hidden rounded-lg border border-slate-700/80 bg-black/60 p-2">
                    {activeDetection?.plate_image_url ? (
                      <img
                        src={mediaUrl(activeDetection.plate_image_url)}
                        alt="Cropped number plate"
                        className="max-h-24 w-full object-contain"
                      />
                    ) : (
                      <span className="text-xs text-slate-500">Plate crop unavailable</span>
                    )}
                  </div>
                  <div className="mt-2 text-[10px] text-slate-500 text-center">
                    Extracted license plate region for OCR
                  </div>
                </div>

                {/* Processed Annotated Image Card */}
                <div className="glass-card flex flex-col rounded-xl p-4 sm:col-span-8">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-[11px] font-bold tracking-wider text-slate-400">
                      PROCESSED IMAGE (BOUNDING BOXES)
                    </div>
                    <button
                      onClick={() => setShowFullAnnotated(true)}
                      className="inline-flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300"
                    >
                      <ZoomIn size={13} />
                      Fullscreen
                    </button>
                  </div>
                  <div
                    onClick={() => setShowFullAnnotated(true)}
                    className="group relative flex min-h-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-slate-700/80 bg-black/60"
                  >
                    <img
                      src={mediaUrl(response.annotated_url || response.image_url)}
                      alt="Processed vehicle detection"
                      className="max-h-[280px] w-full object-contain transition group-hover:scale-[1.01]"
                    />
                    <div className="absolute right-2 top-2 rounded-md bg-black/70 px-2 py-1 text-[10px] text-slate-300 opacity-0 transition group-hover:opacity-100">
                      Click to expand
                    </div>
                  </div>
                  <div className="mt-2.5 flex items-center justify-between text-[10px] text-slate-500">
                    <div className="flex items-center gap-3">
                      <span className="inline-flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-xs border-2 border-cyan-400" />
                        Vehicle Bounding Box
                      </span>
                      <span className="inline-flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-xs border-2 border-emerald-400" />
                        Plate Bounding Box
                      </span>
                    </div>
                    <button
                      onClick={handleDownload}
                      className="font-medium text-cyan-400 hover:text-cyan-300 hover:underline"
                    >
                      Download Image
                    </button>
                  </div>
                </div>
              </div>

              {/* ACTION BUTTONS BAR */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
                <button
                  onClick={handleReset}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-5 py-3 text-xs font-semibold text-slate-200 transition hover:border-cyan-500/40 hover:bg-slate-700 hover:text-white"
                >
                  <RotateCcw size={15} className="text-cyan-400" />
                  Reset &amp; Upload Another Image
                </button>

                <button
                  onClick={handleDownload}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 via-cyan-500 to-blue-500 px-6 py-3 text-xs font-extrabold uppercase tracking-wider text-white shadow-lg shadow-cyan-500/25 transition hover:scale-[1.02] hover:shadow-cyan-500/40"
                >
                  <Download size={15} />
                  Download Processed Image
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* FULLSCREEN ANNOTATED IMAGE MODAL */}
      {showFullAnnotated && response && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 backdrop-blur-md"
          onClick={() => setShowFullAnnotated(false)}
        >
          <div
            className="relative max-h-[92vh] max-w-[94vw] overflow-hidden rounded-2xl border border-cyan-500/30 bg-slate-950 p-2"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setShowFullAnnotated(false)}
              className="absolute right-4 top-4 z-10 grid h-9 w-9 place-items-center rounded-xl bg-black/70 text-slate-300 transition hover:bg-slate-800 hover:text-white"
            >
              <X size={18} />
            </button>
            <img
              src={mediaUrl(response.annotated_url || response.image_url)}
              alt="Annotated detection fullscreen"
              className="max-h-[85vh] w-auto object-contain"
            />
            <div className="mt-2 flex items-center justify-between px-3 py-1 text-xs text-slate-400">
              <span>Annotated YOLO Detection View</span>
              <button
                onClick={handleDownload}
                className="inline-flex items-center gap-1.5 text-cyan-400 hover:text-cyan-300 hover:underline"
              >
                <Download size={13} />
                Download Original File
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ==========================================
// VEHICLE CATEGORY ICON COMPONENT
// ==========================================
function VehicleIcon({ type }: { type: string }) {
  const norm = type.toLowerCase();
  if (norm.includes("truck") || norm === "lorry") {
    return <Truck size={17} className="text-amber-400" />;
  }
  if (norm.includes("bike") || norm === "scooter" || norm === "motorcycle") {
    return <Bike size={17} className="text-purple-400" />;
  }
  if (norm.includes("bus")) {
    return <Bus size={17} className="text-emerald-400" />;
  }
  if (norm.includes("tractor")) {
    return <Tractor size={17} className="text-orange-400" />;
  }
  if (norm.includes("auto") || norm.includes("rickshaw")) {
    return <CarFront size={17} className="text-yellow-400" />;
  }
  if (norm.includes("van")) {
    return <Truck size={17} className="text-blue-400" />;
  }
  return <Car size={17} className="text-cyan-400" />;
}

// ==========================================
// NAVIGATION & HEADER COMPONENTS
// ==========================================
function Sidebar({ active, onChange, mobileOpen, health }: {
  active: string;
  onChange: (value: string) => void;
  mobileOpen: boolean;
  health: Health | null;
}) {
  const ready = Boolean(health?.models.vehicle_model_configured && health?.models.plate_model_configured);

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex w-[264px] flex-col border-r border-slate-800/80 bg-[#090f1d] transition-transform lg:translate-x-0 ${
        mobileOpen ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="flex h-[76px] items-center gap-3 border-b border-slate-800/80 px-6">
        <div className="relative grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 shadow-lg shadow-cyan-950/60">
          <Scan size={22} className="text-white" />
          <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-[#090f1d] bg-emerald-400" />
        </div>
        <div>
          <div className="font-display text-[15px] font-extrabold tracking-wider text-white">
            AI <span className="text-cyan-400">ANPR</span>
          </div>
          <div className="text-[9px] font-semibold tracking-widest text-slate-500">
            VEHICLE DETECTION
          </div>
        </div>
        <button
          className="ml-auto text-slate-500 hover:text-white lg:hidden"
          aria-label="Close navigation"
          onClick={() => onChange(active)}
        >
          <X size={18} />
        </button>
      </div>

      <div className="thin-scrollbar flex-1 overflow-y-auto px-3.5 pt-6">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isSelected = active === item.label;
          return (
            <div key={item.label}>
              {item.section && (
                <div className="mb-2 mt-4 px-3 text-[9px] font-bold tracking-[0.2em] text-slate-500">
                  {item.section}
                </div>
              )}
              <button
                onClick={() => onChange(item.label)}
                className={`group mb-1.5 flex w-full items-center gap-3 rounded-xl px-3.5 py-2.5 text-left text-xs font-semibold transition ${
                  isSelected
                    ? "bg-cyan-500/15 text-cyan-300 shadow-sm border border-cyan-500/25"
                    : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                }`}
              >
                <Icon
                  size={17}
                  strokeWidth={2}
                  className={isSelected ? "text-cyan-400" : "text-slate-500 group-hover:text-slate-300"}
                />
                <span>{item.label}</span>
                {item.label === "AI Detection" && (
                  <span className="ml-auto rounded-full bg-cyan-400/20 px-2 py-0.5 text-[9px] font-bold text-cyan-300">
                    CORE
                  </span>
                )}
                {item.label === "Live detection" && (
                  <span className="ml-auto flex items-center gap-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-[9px] font-bold uppercase text-rose-300">
                    <span className="h-1.5 w-1.5 rounded-full bg-rose-400 soft-pulse" />
                    Live
                  </span>
                )}
              </button>
            </div>
          );
        })}
      </div>

      <div className="mt-auto border-t border-slate-800/80 p-4">
        <div className="rounded-xl border border-cyan-500/20 bg-gradient-to-br from-cyan-500/10 to-blue-600/5 p-3.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
            <Sparkles size={14} className="text-cyan-400" />
            Detection Engine
          </div>
          <p className="mt-1 text-[10px] leading-relaxed text-slate-400">
            Custom YOLO plate weights &amp; YOLO11 vehicle detector loaded.
          </p>
          <div className="mt-2.5 flex items-center gap-2 text-[10px] font-medium text-emerald-300">
            <span className={`h-1.5 w-1.5 rounded-full ${ready ? "bg-emerald-400" : "bg-amber-400"}`} />
            {ready ? "AI Models Loaded & Ready" : "Checking Model Weights..."}
          </div>
        </div>
      </div>
    </aside>
  );
}

function Topbar({ page, onMenu, query, setQuery, health }: {
  page: string;
  onMenu: () => void;
  query: string;
  setQuery: (value: string) => void;
  health: Health | null;
}) {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const ready = Boolean(health?.models.vehicle_model_configured && health?.models.plate_model_configured);
  const status = !health
    ? "API offline"
    : ready
    ? "YOLO11 & Plate Models Active"
    : health.models.plate_model_configured
    ? "Vehicle model missing"
    : "Plate model missing";

  return (
    <header className="sticky top-0 z-20 flex h-[76px] items-center justify-between border-b border-slate-800/80 bg-[#060b14]/90 px-4 backdrop-blur-xl sm:px-7 lg:px-9">
      <div className="flex items-center gap-3">
        <button
          className="text-slate-400 hover:text-white lg:hidden"
          onClick={onMenu}
          aria-label="Open navigation"
        >
          <Menu size={22} />
        </button>
        <div className="hidden text-xs font-semibold tracking-wider text-slate-500 sm:block">
          AI SYSTEM <span className="mx-2 text-slate-700">/</span>{" "}
          <span className="text-slate-200">{page.toUpperCase()}</span>
        </div>
      </div>

      <div className="flex items-center gap-3 sm:gap-5">
        <label className="hidden h-9 w-[220px] items-center gap-2 rounded-xl border border-slate-700/80 bg-slate-900/60 px-3 md:flex">
          <Search size={14} className="shrink-0 text-slate-500" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search plate or vehicle..."
            className="w-full bg-transparent text-xs text-slate-200 outline-none placeholder:text-slate-500"
          />
        </label>

        <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-1.5">
          <span className={`h-2 w-2 rounded-full ${ready ? "bg-emerald-400" : health ? "bg-amber-400" : "bg-rose-400"}`} />
          <span className="text-xs text-slate-300">{status}</span>
        </div>

        <div className="hidden text-right md:block">
          <div className="font-display text-xs font-bold tabular-nums text-slate-200">
            {now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </div>
          <div className="text-[10px] text-slate-500">
            {now.toLocaleDateString([], { weekday: "short", day: "2-digit", month: "short", year: "numeric" })}
          </div>
        </div>
      </div>
    </header>
  );
}

function PageHeading({ eyebrow, title, description, children }: {
  eyebrow: string;
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
      <div>
        <div className="mb-1 text-[10px] font-bold tracking-[0.2em] text-cyan-400/80">{eyebrow}</div>
        <h1 className="font-display text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">{title}</h1>
        <p className="mt-1 text-xs text-slate-400">{description}</p>
      </div>
      {children}
    </div>
  );
}

// ==========================================
// DASHBOARD & ANALYTICS VIEWS
// ==========================================
function Dashboard({ stats, health: _health, loading: _loading, onNavigate, tableProps }: {
  stats: Stats;
  health: Health | null;
  loading: boolean;
  onNavigate: (value: string) => void;
  tableProps: ComponentProps<typeof RecordsTable>;
}) {
  const kpis = [
    { name: "Total Vehicles", value: stats.total_vehicles, note: "Across all captured records", icon: CarFront, color: "cyan" },
    { name: "Plates Recognized", value: stats.plates_recognized, note: "OCR validated registration", icon: ShieldCheck, color: "green" },
    { name: "Cars Detected", value: stats.cars_detected, note: "Classification model", icon: Car, color: "blue" },
    { name: "Bikes Detected", value: stats.bikes_detected, note: "Classification model", icon: Bike, color: "purple" },
    { name: "Trucks Detected", value: stats.trucks_detected, note: "Classification model", icon: Truck, color: "orange" },
    { name: "Recognition Rate", value: `${stats.recognition_rate}%`, note: "Successful OCR / Total", icon: Activity, color: "cyan" },
  ];

  const chartData = useMemo(() => {
    return stats.daily_counts.map((item) => ({
      day: new Date(`${item.date}T00:00:00`).toLocaleDateString([], { weekday: "short" }),
      vehicles: item.count,
    }));
  }, [stats.daily_counts]);

  return (
    <>
      <PageHeading
        eyebrow="WORKSPACE / DASHBOARD"
        title="Detection Dashboard"
        description="Overview of traffic volume, classification statistics, and real-time ANPR performance."
      >
        <button
          onClick={() => onNavigate("AI Detection")}
          className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-4 py-2.5 text-xs font-bold text-white shadow-lg shadow-cyan-500/20"
        >
          <Scan size={15} />
          Open AI Detection Studio
        </button>
      </PageHeading>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        {kpis.map((kpi) => {
          const Icon = kpi.icon;
          return (
            <div key={kpi.name} className="glass-card rounded-xl p-4">
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>{kpi.name}</span>
                <Icon size={16} className="text-cyan-400" />
              </div>
              <div className="mt-2 font-display text-xl font-extrabold text-white sm:text-2xl">
                {kpi.value}
              </div>
              <div className="mt-1 text-[10px] text-slate-500 truncate">{kpi.note}</div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-12">
        <div className="glass-card rounded-2xl p-5 lg:col-span-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-white">Daily Traffic Volume</h2>
              <p className="text-xs text-slate-500">Total detected vehicles over the past 7 days</p>
            </div>
            <span className="rounded-md border border-cyan-500/20 bg-cyan-500/10 px-2 py-0.5 text-[10px] font-semibold text-cyan-300">
              7 DAYS
            </span>
          </div>
          <div className="mt-4 h-[250px]">
            {chartData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorVehicles" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#ffffff0d" vertical={false} />
                  <XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#64748b", fontSize: 10 }} />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: "#64748b", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #38bdf820", borderRadius: 8, color: "#fff", fontSize: 11 }} />
                  <Area type="monotone" dataKey="vehicles" stroke="#38bdf8" strokeWidth={2} fillOpacity={1} fill="url(#colorVehicles)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-slate-500">
                No activity data recorded yet
              </div>
            )}
          </div>
        </div>

        <div className="glass-card rounded-2xl p-5 lg:col-span-4">
          <h2 className="font-display text-sm font-bold text-white">Model Status</h2>
          <p className="text-xs text-slate-500">Current AI pipeline readiness</p>

          <div className="mt-4 space-y-3">
            <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-black/30 p-3 text-xs">
              <span className="text-slate-400">Vehicle Model</span>
              <span className="font-semibold text-emerald-400">YOLO11 Ready</span>
            </div>
            <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-black/30 p-3 text-xs">
              <span className="text-slate-400">Plate Model</span>
              <span className="font-semibold text-emerald-400">Plate YOLO Ready</span>
            </div>
            <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-black/30 p-3 text-xs">
              <span className="text-slate-400">OCR Engine</span>
              <span className="font-semibold text-cyan-400">EasyOCR Active</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6">
        <section className="glass-card overflow-hidden rounded-2xl">
          <div className="border-b border-slate-800 px-5 py-4">
            <h2 className="font-display text-sm font-bold text-white">Recent Vehicle Records</h2>
            <p className="text-xs text-slate-500">Live detection database entries</p>
          </div>
          <RecordsTable {...tableProps} compact={true} />
        </section>
      </div>
    </>
  );
}

// ==========================================
// RECORDS & TABLES
// ==========================================
type TableProps = {
  detections: Detection[];
  total: number;
  pageIndex: number;
  onPage: (page: number) => void;
  onSelect: (item: Detection) => void;
  onExport: () => void;
  query: string;
  setQuery: (value: string) => void;
  vehicleFilter: string;
  setVehicleFilter: (value: string) => void;
  fromDate: string;
  setFromDate: (value: string) => void;
  toDate: string;
  setToDate: (value: string) => void;
};

function RecordsPage(props: TableProps & { title: string }) {
  return (
    <>
      <PageHeading
        eyebrow="INTELLIGENCE / RECORDS"
        title={props.title === "Plate search" ? "Number Plate Search" : "Vehicle Detection Records"}
        description="Search, filter, inspect, and export actual license plate detections stored in the local SQLite database."
      />
      <section className="glass-card overflow-hidden rounded-2xl">
        <RecordsTable {...props} />
      </section>
    </>
  );
}

function RecordsTable({
  detections, total, pageIndex, onPage, onSelect, onExport,
  query, setQuery, vehicleFilter, setVehicleFilter,
  fromDate, setFromDate, toDate, setToDate, compact = false,
}: TableProps & { compact?: boolean }) {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const pages = Math.max(1, Math.ceil(total / 10));

  return (
    <>
      {!compact && (
        <div className="flex flex-col gap-3 border-b border-slate-800/80 p-4 sm:flex-row sm:items-center sm:justify-between">
          <label className="flex h-9 w-full items-center gap-2 rounded-xl border border-slate-700/80 bg-slate-900/60 px-3 sm:max-w-[280px]">
            <Search size={14} className="text-slate-500" />
            <input
              aria-label="Search registration"
              value={query}
              onChange={(e) => { setQuery(e.target.value); onPage(1); }}
              placeholder="Search registration or category..."
              className="w-full bg-transparent text-xs text-slate-200 outline-none placeholder:text-slate-500"
            />
          </label>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setFiltersOpen(!filtersOpen)}
              className={`inline-flex h-9 items-center gap-2 rounded-xl border px-3 text-xs font-semibold ${
                filtersOpen ? "border-cyan-400/40 text-cyan-300" : "border-slate-700 text-slate-400 hover:text-white"
              }`}
            >
              <SlidersHorizontal size={13} /> Filters
            </button>
            <button
              onClick={onExport}
              className="inline-flex h-9 items-center gap-2 rounded-xl border border-slate-700 px-3 text-xs font-semibold text-slate-300 transition hover:border-cyan-400/40 hover:text-cyan-300"
            >
              <Download size={13} /> Export CSV
            </button>
          </div>
        </div>
      )}

      {filtersOpen && !compact && (
        <div className="grid gap-3 border-b border-slate-800/80 bg-slate-900/40 p-4 sm:grid-cols-3">
          <label className="text-[10px] font-bold text-slate-400">
            VEHICLE CATEGORY
            <select
              value={vehicleFilter}
              onChange={(e) => { setVehicleFilter(e.target.value); onPage(1); }}
              className="mt-1.5 block h-9 w-full rounded-lg border border-slate-700 bg-slate-900 px-2.5 text-xs text-slate-200 outline-none"
            >
              <option value="">All Categories</option>
              {["Car", "Bike", "Truck", "Bus", "Auto", "Van", "Tractor"].map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </label>
          <label className="text-[10px] font-bold text-slate-400">
            FROM DATE
            <input
              type="date"
              value={fromDate}
              onChange={(e) => { setFromDate(e.target.value); onPage(1); }}
              className="mt-1.5 block h-9 w-full rounded-lg border border-slate-700 bg-slate-900 px-2.5 text-xs text-slate-200 outline-none"
            />
          </label>
          <label className="text-[10px] font-bold text-slate-400">
            TO DATE
            <input
              type="date"
              value={toDate}
              onChange={(e) => { setToDate(e.target.value); onPage(1); }}
              className="mt-1.5 block h-9 w-full rounded-lg border border-slate-700 bg-slate-900 px-2.5 text-xs text-slate-200 outline-none"
            />
          </label>
        </div>
      )}

      <div className="thin-scrollbar overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/40 text-[9px] font-bold uppercase tracking-wider text-slate-400">
              <th className="px-5 py-3">FRAME</th>
              <th className="px-4 py-3">CATEGORY</th>
              <th className="px-4 py-3">REGISTRATION NUMBER</th>
              <th className="px-4 py-3">TIMESTAMP</th>
              <th className="px-4 py-3">CONFIDENCE</th>
              <th className="px-4 py-3">STATUS</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {detections.map((item) => (
              <tr
                key={item.id}
                onClick={() => onSelect(item)}
                className="group cursor-pointer border-b border-slate-800/60 transition hover:bg-slate-800/30"
              >
                <td className="px-5 py-3">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-16 overflow-hidden rounded-lg border border-slate-700 bg-slate-950">
                      {item.annotated_url ? (
                        <img src={mediaUrl(item.annotated_url)} alt="Detection" className="h-full w-full object-cover" />
                      ) : (
                        <div className="grid h-full place-items-center text-slate-600">
                          <CarFront size={16} />
                        </div>
                      )}
                    </div>
                    <span className="font-mono text-xs text-slate-400">#{item.id}</span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/60 px-2.5 py-1 text-xs font-semibold text-slate-300">
                    <VehicleIcon type={item.vehicle_type} />
                    <span>{item.vehicle_type}</span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className={`font-mono text-xs font-bold tracking-wider ${item.plate_text ? "text-cyan-300" : "text-slate-500"}`}>
                    {item.plate_text || "Not Recognized"}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="text-xs text-slate-300">{formatDate(item.created_at)}</div>
                  <div className="text-[10px] text-slate-500">{formatTime(item.created_at)}</div>
                </td>
                <td className="px-4 py-3">
                  <div className="text-xs text-slate-300">{Math.round(item.vehicle_confidence * 100)}% vehicle</div>
                  <div className="text-[10px] text-slate-500">
                    {item.ocr_confidence ? `${Math.round(item.ocr_confidence * 100)}% OCR` : "— OCR"}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={item.status} />
                </td>
                <td className="px-4 py-3">
                  <ChevronRight size={14} className="text-slate-600 transition group-hover:translate-x-1 group-hover:text-cyan-400" />
                </td>
              </tr>
            ))}

            {!detections.length && (
              <tr>
                <td colSpan={7}>
                  <div className="flex flex-col items-center py-12 text-center text-slate-500">
                    <Search size={22} className="mb-2 text-slate-600" />
                    <div className="text-sm font-semibold text-slate-400">No detection records found</div>
                    <div className="mt-1 text-xs">Run a detection from the AI Detection Studio to see live results here.</div>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between border-t border-slate-800/80 px-5 py-3 text-xs text-slate-500">
        <span>Showing {total ? (pageIndex - 1) * 10 + 1 : 0}–{Math.min(pageIndex * 10, total)} of {total} detections</span>
        <div className="flex items-center gap-1.5">
          <button
            disabled={pageIndex <= 1}
            onClick={() => onPage(pageIndex - 1)}
            className="grid h-8 w-8 place-items-center rounded-lg border border-slate-700 disabled:opacity-30"
          >
            <ChevronLeft size={14} />
          </button>
          <span className="px-2 text-slate-300">{pageIndex} / {pages}</span>
          <button
            disabled={pageIndex >= pages}
            onClick={() => onPage(pageIndex + 1)}
            className="grid h-8 w-8 place-items-center rounded-lg border border-slate-700 disabled:opacity-30"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "recognized") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-2.5 py-0.5 text-[10px] font-bold text-emerald-300">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        Recognized
      </span>
    );
  }
  if (status === "low_confidence") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/25 bg-amber-400/10 px-2.5 py-0.5 text-[10px] font-bold text-amber-300">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
        Verify Plate
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-400/25 bg-rose-400/10 px-2.5 py-0.5 text-[10px] font-bold text-rose-300">
      <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
      Plate Not Found
    </span>
  );
}

// ==========================================
// VIDEO & LIVE CAMERA ANALYTICS
// ==========================================
function UploadPage({ kind: _kind, onComplete }: { kind: "video"; onComplete: () => void }) {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const onUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploading(true);
    setError("");
    setMessage("");

    const data = new FormData();
    data.append("file", file);

    try {
      const result = await request<{ detections: Detection[]; message?: string }>("/api/detections/video", {
        method: "POST",
        body: data,
      });
      setMessage(result.message || `Video analysis complete — ${result.detections.length} vehicles detected.`);
      onComplete();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Video analysis failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <PageHeading
        eyebrow="ANALYSIS / VIDEO"
        title="Video Stream Analysis"
        description="Analyze video files (MP4, MOV, AVI) by sampling frames and recording detected vehicles and license plates."
      />
      <div className="glass-card max-w-2xl rounded-2xl p-7">
        <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={onUpload} />
        <button
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
          className="flex min-h-[260px] w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-900/40 p-6 text-center transition hover:border-cyan-400 hover:bg-cyan-500/5 disabled:opacity-50"
        >
          <Video size={34} className="mb-3 text-cyan-400" />
          <div className="font-display text-base font-bold text-slate-100">
            {uploading ? "Sampling & Analyzing Frames..." : "Select Video File"}
          </div>
          <p className="mt-1 text-xs text-slate-500">Supports MP4, MOV, AVI. Analyzes 1 in every 10 frames.</p>
        </button>

        {error && <div className="mt-4 rounded-xl border border-rose-500/25 bg-rose-500/10 p-3 text-xs text-rose-200">{error}</div>}
        {message && <div className="mt-4 rounded-xl border border-emerald-500/25 bg-emerald-500/10 p-3 text-xs text-emerald-200">{message}</div>}
      </div>
    </>
  );
}

function LiveDetection({ health: _health }: { health: Health | null }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const [active, setActive] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [message, setMessage] = useState("");
  const [detections, setDetections] = useState<Array<{ vehicle_type: string; vehicle_confidence: number; plate_text: string | null; ocr_confidence: number | null; status: string; vehicle_box: number[]; plate_box: number[] | null }>>([]);
  const [timestamp, setTimestamp] = useState("");

  const stop = useCallback(() => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
    setConnecting(false);
  }, []);

  useEffect(() => () => stop(), [stop]);

  const start = async () => {
    setMessage("");
    setConnecting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const socket = new WebSocket(`${WS}/api/ws/live`);
      socketRef.current = socket;
      socket.onopen = () => {
        setActive(true);
        setConnecting(false);
        timerRef.current = window.setInterval(() => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (!video || !canvas || video.readyState < 2 || socket.readyState !== WebSocket.OPEN) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const context = canvas.getContext("2d");
          if (!context) return;
          context.drawImage(video, 0, 0);
          socket.send(JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.7).split(",")[1] }));
        }, 1600);
      };
      socket.onmessage = (event) => {
        const result = JSON.parse(event.data);
        if (result.error) { setMessage(result.error); return; }
        setDetections(result.detections || []);
        setTimestamp(result.timestamp || "");
      };
      socket.onerror = () => setMessage("Live connection failed. Check backend.");
      socket.onclose = () => setActive(false);
    } catch (reason) {
      stop();
      setMessage(reason instanceof Error ? reason.message : "Could not access camera.");
    }
  };

  return (
    <>
      <PageHeading
        eyebrow="MONITORING / LIVE"
        title="Live Camera Feed"
        description="Stream camera snapshots to the backend YOLO and OCR pipeline in real-time."
      >
        <button
          onClick={active ? stop : start}
          disabled={connecting}
          className={`rounded-xl px-5 py-2.5 text-xs font-bold ${
            active ? "border border-rose-500/30 bg-rose-500/10 text-rose-300" : "bg-gradient-to-r from-blue-600 to-cyan-500 text-white"
          }`}
        >
          {connecting ? "Connecting..." : active ? "Stop Camera" : "Start Camera"}
        </button>
      </PageHeading>

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="glass-card overflow-hidden rounded-2xl lg:col-span-8">
          <div className="relative aspect-video bg-black flex items-center justify-center">
            <video ref={videoRef} muted playsInline className={`h-full w-full object-contain ${active ? "" : "hidden"}`} />
            {!active && (
              <div className="flex flex-col items-center text-center text-slate-500">
                <Camera size={36} className="mb-2 text-slate-600" />
                <div className="text-sm font-semibold">Camera Offline</div>
              </div>
            )}
            <canvas ref={canvasRef} className="hidden" />
          </div>
        </div>

        <div className="glass-card rounded-2xl p-5 lg:col-span-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-sm font-bold text-white">Live Snapshots</h2>
            {timestamp && <span className="text-[10px] text-slate-400 font-mono">{formatTime(timestamp)}</span>}
          </div>
          {message && <div className="mt-3 rounded-lg border border-rose-500/25 bg-rose-500/10 p-2.5 text-xs text-rose-200">{message}</div>}
          <div className="mt-4 space-y-2.5">
            {detections.map((d, i) => (
              <div key={i} className="rounded-xl border border-slate-800 bg-black/40 p-3">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                  <span>{d.vehicle_type}</span>
                  <StatusBadge status={d.status} />
                </div>
                <div className="mt-2 font-mono text-base font-bold text-cyan-300">
                  {d.plate_text || "Uncertain Reading"}
                </div>
              </div>
            ))}
            {!detections.length && !message && (
              <div className="py-8 text-center text-xs text-slate-500">Awaiting detection frames...</div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

// ==========================================
// ANALYTICS & REPORTS
// ==========================================
function Analytics({ stats }: { stats: Stats }) {
  const distribution = Object.entries(stats.by_vehicle_type).map(([name, value]) => ({ name, value }));

  return (
    <>
      <PageHeading
        eyebrow="INTELLIGENCE / ANALYTICS"
        title="Traffic Analytics"
        description="Distribution of vehicle classes and recognition rate from saved detection records."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="glass-card rounded-2xl p-6">
          <h2 className="font-display text-sm font-bold text-white">Vehicle Category Breakdown</h2>
          <div className="mt-4 h-[280px]">
            {distribution.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={distribution} dataKey="value" nameKey="name" innerRadius={60} outerRadius={90} paddingAngle={4}>
                    {distribution.map((item, i) => (
                      <Cell key={item.name} fill={chartColors[i % chartColors.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #38bdf820", borderRadius: 8, color: "#fff" }} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-slate-500">No category data yet</div>
            )}
          </div>
        </div>

        <div className="glass-card rounded-2xl p-6">
          <h2 className="font-display text-sm font-bold text-white">Recognition Success Ratio</h2>
          <div className="mt-6">
            <div className="flex justify-between text-xs text-slate-400">
              <span>Overall Plate Recognition Rate</span>
              <span className="font-bold text-cyan-400">{stats.recognition_rate}%</span>
            </div>
            <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-gradient-to-r from-blue-600 to-cyan-400" style={{ width: `${stats.recognition_rate}%` }} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Reports({ stats, detections: _detections, onExport }: { stats: Stats; detections: Detection[]; onExport: () => void }) {
  return (
    <>
      <PageHeading
        eyebrow="MANAGEMENT / REPORTS"
        title="Detection Reports &amp; Data Export"
        description="Export detection histories or review executive summary metrics."
      />
      <div className="glass-card max-w-xl rounded-2xl p-6">
        <div className="font-display text-base font-bold text-white">Export Detection Dataset</div>
        <p className="mt-1 text-xs text-slate-400">
          Download a complete CSV dataset of detected vehicle types, extracted registration numbers, OCR confidences, and timestamps.
        </p>
        <div className="mt-5 flex items-center gap-4">
          <button
            onClick={onExport}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-5 py-2.5 text-xs font-bold text-white shadow-lg shadow-cyan-500/20"
          >
            <Download size={14} />
            Download CSV Export
          </button>
          <span className="text-xs text-slate-500">{stats.total_vehicles} total records available</span>
        </div>
      </div>
    </>
  );
}

function SettingsPage({ health }: { health: Health | null }) {
  return (
    <>
      <PageHeading
        eyebrow="SYSTEM / SETTINGS"
        title="Model &amp; System Configuration"
        description="Review backend AI model file paths and active inference configurations."
      />
      <div className="glass-card max-w-2xl rounded-2xl p-6">
        <h2 className="font-display text-sm font-bold text-white">Loaded AI Models</h2>
        <div className="mt-4 space-y-3 text-xs">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <span className="text-slate-400">Backend API URL</span>
            <span className="font-mono text-slate-200">{API}</span>
          </div>
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <span className="text-slate-400">Vehicle Model (YOLO11)</span>
            <span className="font-semibold text-emerald-400">
              {health?.models.vehicle_model_configured ? "Active · yolo11n.pt" : "Missing"}
            </span>
          </div>
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <span className="text-slate-400">Plate Detector Model</span>
            <span className="font-semibold text-emerald-400">
              {health?.models.plate_model_configured ? "Active · plate_model.pt" : "Missing"}
            </span>
          </div>
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
            <span className="text-slate-400">OCR Engine</span>
            <span className="font-semibold text-cyan-400">EasyOCR (PyTorch CPU/CUDA)</span>
          </div>
        </div>
      </div>
    </>
  );
}

function DetectionDetail({ detection, onClose }: { detection: Detection; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm" onClick={onClose}>
      <div
        className="glass-card thin-scrollbar max-h-[90vh] w-full max-w-[700px] overflow-y-auto rounded-2xl p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <h2 className="font-display text-base font-bold text-white">Detection #{detection.id} Details</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={18} /></button>
        </div>
        <div className="mt-4 space-y-4 text-xs">
          <div className="grid grid-cols-2 gap-3">
            {detection.image_url && (
              <div>
                <div className="mb-1 text-[10px] text-slate-400">ORIGINAL</div>
                <img src={mediaUrl(detection.image_url)} alt="Original" className="h-36 w-full rounded-lg object-contain bg-black" />
              </div>
            )}
            {detection.annotated_url && (
              <div>
                <div className="mb-1 text-[10px] text-slate-400">ANNOTATED</div>
                <img src={mediaUrl(detection.annotated_url)} alt="Annotated" className="h-36 w-full rounded-lg object-contain bg-black" />
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-300">{detection.vehicle_type}</span>
              <StatusBadge status={detection.status} />
            </div>
            <div className="mt-3 font-mono text-2xl font-black text-cyan-300">
              {detection.plate_text || "Manual Verification Required"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
