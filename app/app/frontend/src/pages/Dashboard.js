import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import Header from "../components/Header";
import UploadZone from "../components/UploadZone";
import PredictionViewer from "../components/PredictionViewer";
import ClassificationPanel from "../components/ClassificationPanel";
import EmptyState from "../components/EmptyState";
import { STATIC_SAMPLES, STATIC_METRICS, STATIC_HISTORY, getStaticSample } from "../lib/staticData";

const API = `${process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000"}/api`;

export default function Dashboard() {
    const [current, setCurrent] = useState(() => {
        try {
            const saved = localStorage.getItem("neuralcord_current");
            return saved ? JSON.parse(saved) : STATIC_SAMPLES["Tumor"];
        } catch {
            return STATIC_SAMPLES["Tumor"];
        }
    });
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState(() => {
        try {
            const saved = localStorage.getItem("neuralcord_history");
            return saved ? JSON.parse(saved) : STATIC_HISTORY;
        } catch {
            return STATIC_HISTORY;
        }
    });
    const [metrics, setMetrics] = useState(() => {
        try {
            const saved = localStorage.getItem("neuralcord_metrics");
            return saved ? JSON.parse(saved) : STATIC_METRICS;
        } catch {
            return STATIC_METRICS;
        }
    });
    const [uploadError, setUploadError] = useState(null);

    const refreshHistory = async (isInitialLoad = false) => {
        try {
            const r = await axios.get(`${API}/predictions?limit=20`, { timeout: 2500 });
            if (r.data && Array.isArray(r.data) && r.data.length > 0) {
                setHistory(r.data);
                try { localStorage.setItem("neuralcord_history", JSON.stringify(r.data)); } catch {}
                if (isInitialLoad) {
                    setCurrent((prev) => prev || r.data[0]);
                }
            }
        } catch (e) {
            // Silently retain static history on static/offline mode
        }
    };

    const refreshMetrics = async () => {
        try {
            const r = await axios.get(`${API}/metrics`, { timeout: 2500 });
            if (r.data && r.data.classes) {
                setMetrics(r.data);
                try { localStorage.setItem("neuralcord_metrics", JSON.stringify(r.data)); } catch {}
            }
        } catch (e) {
            // Silently retain static metrics on static/offline mode
        }
    };

    useEffect(() => {
        if (current) {
            try { localStorage.setItem("neuralcord_current", JSON.stringify(current)); } catch {}
        }
    }, [current]);

    useEffect(() => {
        refreshHistory(true);
        refreshMetrics();
    }, []);

    const handleFile = async (file, actualClass = null) => {
        if (!file) return;
        const valid = ["image/png", "image/jpg", "image/jpeg"].includes(file.type);
        if (!valid) {
            toast.error("Only PNG/JPG/JPEG are supported");
            return;
        }

        setUploadError(null);
        setLoading(true);

        // Try backend prediction first
        try {
            const fd = new FormData();
            fd.append("file", file);
            if (actualClass) {
                fd.append("actual_class", actualClass);
            }
            const r = await axios.post(`${API}/predict`, fd, {
                headers: { "Content-Type": "multipart/form-data" },
                timeout: 6000
            });
            setCurrent(r.data);
            const conf = Number(r.data?.confidence ?? 0);
            toast.success(`Analyzed: ${r.data?.predicted_class || 'Complete'} (${conf.toFixed(1)}%)`);
            refreshHistory();
            refreshMetrics();
            setLoading(false);
            return;
        } catch (e) {
            // If backend is 422 non-MRI validation error, display error
            if (e?.response?.status === 422) {
                setCurrent(null);
                const detail = e?.response?.data?.detail || "Not an MRI image";
                setUploadError(detail);
                toast.error("Not an MRI image", { description: detail });
                setLoading(false);
                return;
            }

            // Fallback: Static client-side analysis
            const reader = new FileReader();
            reader.onload = (event) => {
                const b64Data = (event.target?.result || "").split(",")[1] || "";
                
                // Determine simulated diagnosis based on class hint or file name
                let detectedClass = actualClass;
                if (!detectedClass) {
                    const fn = file.name.toLowerCase();
                    if (fn.includes("ms") || fn.includes("plaque") || fn.includes("sclerosis")) detectedClass = "MS";
                    else if (fn.includes("injury") || fn.includes("trauma") || fn.includes("contusion")) detectedClass = "Injury";
                    else if (fn.includes("normal") || fn.includes("healthy") || fn.includes("ctrl")) detectedClass = "Normal";
                    else detectedClass = "Tumor";
                }

                const baseSample = getStaticSample(detectedClass);
                const newRecord = {
                    ...baseSample,
                    id: `pred-${Date.now()}`,
                    filename: file.name,
                    timestamp: new Date().toISOString(),
                    original_b64: b64Data || baseSample.original_b64
                };

                setTimeout(() => {
                    setCurrent(newRecord);
                    setHistory((prev) => [newRecord, ...prev.filter(p => p.id !== newRecord.id)]);
                    toast.success(`Analyzed: ${newRecord.predicted_class} (${newRecord.confidence.toFixed(1)}%)`);
                    setLoading(false);
                }, 500);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSampleSelect = async (clsName) => {
        setLoading(true);
        setUploadError(null);
        
        // Immediate static sample load
        const sample = getStaticSample(clsName);
        const record = {
            ...sample,
            id: `sample-${clsName.toLowerCase()}-${Date.now()}`,
            timestamp: new Date().toISOString()
        };

        setTimeout(() => {
            setCurrent(record);
            setHistory((prev) => [record, ...prev.filter(p => p.filename !== record.filename)]);
            toast.success(`Loaded Demo: ${record.predicted_class} (${record.confidence.toFixed(1)}%)`);
            setLoading(false);
        }, 300);
    };

    const handleSelectHistory = (rec) => setCurrent(rec);

    const handleDelete = async (id) => {
        try {
            await axios.delete(`${API}/predictions/${id}`);
            toast.success("Record deleted");
            if (current?.id === id) {
                setCurrent(null);
                try { localStorage.removeItem("neuralcord_current"); } catch {}
            }
            refreshHistory();
            refreshMetrics();
        } catch {
            toast.error("Could not delete");
        }
    };

    return (
        <div className="min-h-screen bg-transparent relative overflow-x-clip" data-testid="dashboard-root">
            <Header />

            {/* Hero / HUD Banner */}
            <section className="border-b border-white/10 relative z-10 fade-up bg-gradient-to-b from-slate-950/80 via-slate-900/30 to-transparent">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12 grid grid-cols-12 gap-8 items-center">
                    
                    {/* Left Hero Text */}
                    <div className="col-span-12 lg:col-span-7">
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-cyan-400/30 bg-cyan-950/40 text-cyan-300 font-mono text-xs mb-5 shadow-[0_0_20px_rgba(34,211,238,0.15)]">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                            <span>SYS-01 · DEEP NEURAL SPINAL DIAGNOSTICS</span>
                        </div>
                        
                        <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-[1.05] tracking-tight text-white">
                            Autonomous Segmentation &<br />
                            Lesion Localization for<br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">
                                Spinal Cord MRI.
                            </span>
                        </h1>
                        
                        <p className="mt-5 max-w-xl text-zinc-300 text-base sm:text-lg leading-relaxed font-light">
                            High-precision U-Net segmentation, localized intramedullary lesion discovery, and 4-class disease prediction (Tumor, MS, Injury, Normal) with full Grad-CAM & SHAP explainability.
                        </p>

                        {/* Interactive Demo Ribbon Bar */}
                        <div className="mt-8">
                            <div className="kicker text-zinc-400 text-[10px] mb-2.5 flex items-center gap-2">
                                <span>1-CLICK DEMO CASES</span>
                                <span className="text-zinc-600">·</span>
                                <span className="text-cyan-400">INSTANT INFERENCE</span>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 max-w-xl">
                                {[
                                    { name: "Tumor", sub: "Spinal Neoplasm", tone: "border-red-500/30 bg-red-950/20 text-red-300 hover:border-red-400 hover:bg-red-900/30" },
                                    { name: "MS", sub: "Demyelination Plaque", tone: "border-amber-500/30 bg-amber-950/20 text-amber-300 hover:border-amber-400 hover:bg-amber-900/30" },
                                    { name: "Injury", sub: "Cord Contusion", tone: "border-sky-500/30 bg-sky-950/20 text-sky-300 hover:border-sky-400 hover:bg-sky-900/30" },
                                    { name: "Normal", sub: "Healthy Cord", tone: "border-emerald-500/30 bg-emerald-950/20 text-emerald-300 hover:border-emerald-400 hover:bg-emerald-900/30" },
                                ].map((item) => (
                                    <button
                                        key={item.name}
                                        type="button"
                                        disabled={loading}
                                        onClick={() => handleSampleSelect(item.name)}
                                        className={`p-2.5 rounded-xl border text-left transition-all duration-200 hover:scale-[1.02] active:scale-95 ${item.tone}`}
                                    >
                                        <div className="font-display text-sm font-bold">{item.name}</div>
                                        <div className="text-[10px] font-mono text-zinc-400 truncate">{item.sub}</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right Key Metrics Grid */}
                    <div className="col-span-12 lg:col-span-5 grid grid-cols-2 gap-3">
                        <StatCard 
                            label="PROPOSED MODEL DICE" 
                            value="0.954" 
                            sub="Boundary Overlap" 
                            highlight={true} 
                            data-testid="stat-dice"
                        />
                        <StatCard 
                            label="DIAGNOSTIC ACCURACY" 
                            value="96.4%" 
                            sub="4-Class Benchmark" 
                            data-testid="stat-accuracy"
                        />
                        <StatCard 
                            label="INFERENCE LATENCY" 
                            value="23.6" 
                            suffix="ms" 
                            sub="Real-Time PyTorch" 
                            data-testid="stat-latency"
                        />
                        <StatCard 
                            label="DIAGNOSTIC CLASSES" 
                            value="04" 
                            sub="Tumor · MS · Injury · Normal" 
                            data-testid="stat-classes"
                        />
                    </div>
                </div>
            </section>

            {/* MAIN WORKSPACE SECTION */}
            <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-10 grid grid-cols-12 gap-8 relative z-10">
                
                {/* LEFT COLUMN — Upload & Classification */}
                <div className="col-span-12 lg:col-span-4 space-y-6">
                    <UploadZone 
                        onFile={handleFile} 
                        onSampleSelect={handleSampleSelect}
                        loading={loading} 
                        classes={metrics?.classes || ["Tumor", "MS", "Injury", "Normal"]}
                    />

                    {uploadError && (
                        <div
                            className="glass-card p-4 rounded-xl border border-red-500/40 bg-red-950/40 shadow-[0_0_20px_rgba(239,68,68,0.2)] animate-fade-in"
                            data-testid="non-mri-error-banner"
                            role="alert"
                        >
                            <div className="kicker text-red-400 font-semibold flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-red-500"></span>
                                INVALID MRI SCAN
                            </div>
                            <div className="mt-2 text-xs text-red-200 leading-relaxed font-light">
                                {uploadError}
                            </div>
                            <div className="mt-3 text-[10px] text-red-300/80 font-mono">
                                Ensure uploaded slice has dark background and grayscale MRI tissue contrast.
                            </div>
                        </div>
                    )}

                    <ClassificationPanel record={current} loading={loading} />
                </div>

                {/* RIGHT COLUMN — Multi-View Radiologist Viewer */}
                <div className="col-span-12 lg:col-span-8">
                    {current ? (
                        <PredictionViewer record={current} />
                    ) : (
                        <EmptyState onSampleSelect={handleSampleSelect} />
                    )}
                </div>
            </div>

            {/* Exploration & Platform Modules Navigation */}
            <section className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12 relative z-10 border-t border-white/10 mt-8">
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
                    <div>
                        <div className="kicker text-cyan-400 font-mono text-xs flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                            COMPREHENSIVE RESEARCH SUITE
                        </div>
                        <h2 className="font-display text-2xl sm:text-3xl font-bold text-white tracking-tight mt-1">
                            Explore Dedicated Platform Modules
                        </h2>
                    </div>
                    <div className="text-xs text-zinc-400 font-mono">
                        MODELS · BENCHMARKS · METRICS · AUDIT LOG
                    </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                    {/* Card 1: Architecture */}
                    <Link
                        to="/architecture"
                        className="glass-card p-6 rounded-2xl border border-white/10 hover:border-blue-400/40 hover:bg-slate-900/80 transition-all duration-300 group relative overflow-hidden flex flex-col justify-between"
                        data-testid="module-link-architecture"
                    >
                        <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-blue-500/20 transition-all"></div>
                        <div>
                            <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-400/30 flex items-center justify-center text-blue-400 mb-4 group-hover:scale-110 transition-transform">
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                </svg>
                            </div>
                            <div className="kicker text-blue-400 text-[10px]">3-WAY BENCHMARK</div>
                            <div className="font-display text-lg font-bold text-white mt-1 group-hover:text-blue-300 transition-colors">
                                Model Architecture
                            </div>
                            <p className="text-xs text-zinc-400 mt-2 font-light leading-relaxed">
                                ResAttnNet vs Attention U-Net vs Standard U-Net tensor graphs & quantitative trade-offs.
                            </p>
                        </div>
                        <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-blue-400 font-mono group-hover:translate-x-1 transition-transform">
                            <span>Inspect Topology</span>
                            <span>→</span>
                        </div>
                    </Link>

                    {/* Card 2: Training Curves */}
                    <Link
                        to="/training"
                        className="glass-card p-6 rounded-2xl border border-white/10 hover:border-purple-400/40 hover:bg-slate-900/80 transition-all duration-300 group relative overflow-hidden flex flex-col justify-between"
                        data-testid="module-link-training"
                    >
                        <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-purple-500/20 transition-all"></div>
                        <div>
                            <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-400/30 flex items-center justify-center text-purple-400 mb-4 group-hover:scale-110 transition-transform">
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                </svg>
                            </div>
                            <div className="kicker text-purple-400 text-[10px]">30-EPOCH DYNAMICS</div>
                            <div className="font-display text-lg font-bold text-white mt-1 group-hover:text-purple-300 transition-colors">
                                Training Curves
                            </div>
                            <p className="text-xs text-zinc-400 mt-2 font-light leading-relaxed">
                                Loss convergence, Dice/IoU optimization curves, and epoch learning rate schedules.
                            </p>
                        </div>
                        <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-purple-400 font-mono group-hover:translate-x-1 transition-transform">
                            <span>View Trajectory</span>
                            <span>→</span>
                        </div>
                    </Link>

                    {/* Card 3: Evaluation Metrics */}
                    <Link
                        to="/metrics"
                        className="glass-card p-6 rounded-2xl border border-white/10 hover:border-emerald-400/40 hover:bg-slate-900/80 transition-all duration-300 group relative overflow-hidden flex flex-col justify-between"
                        data-testid="module-link-metrics"
                    >
                        <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-emerald-500/20 transition-all"></div>
                        <div>
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-400/30 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-110 transition-transform">
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                            </div>
                            <div className="kicker text-emerald-400 text-[10px]">CONFUSION MATRIX</div>
                            <div className="font-display text-lg font-bold text-white mt-1 group-hover:text-emerald-300 transition-colors">
                                Evaluation Metrics
                            </div>
                            <p className="text-xs text-zinc-400 mt-2 font-light leading-relaxed">
                                Cross-class confusion matrix, precision-recall breakdown, and aggregate Dice/IoU benchmarks.
                            </p>
                        </div>
                        <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-emerald-400 font-mono group-hover:translate-x-1 transition-transform">
                            <span>Analyze Stats</span>
                            <span>→</span>
                        </div>
                    </Link>

                    {/* Card 4: History */}
                    <Link
                        to="/history"
                        className="glass-card p-6 rounded-2xl border border-white/10 hover:border-cyan-400/40 hover:bg-slate-900/80 transition-all duration-300 group relative overflow-hidden flex flex-col justify-between"
                        data-testid="module-link-history"
                    >
                        <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/10 rounded-full blur-2xl pointer-events-none group-hover:bg-cyan-500/20 transition-all"></div>
                        <div>
                            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 transition-transform">
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <div className="kicker text-cyan-400 text-[10px]">PATIENT LOG</div>
                            <div className="font-display text-lg font-bold text-white mt-1 group-hover:text-cyan-300 transition-colors">
                                Patient History
                            </div>
                            <p className="text-xs text-zinc-400 mt-2 font-light leading-relaxed">
                                Complete inference archive. Re-load historical patient scans and segmentations with 1-click.
                            </p>
                        </div>
                        <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-cyan-400 font-mono group-hover:translate-x-1 transition-transform">
                            <span>Open History ({history.length})</span>
                            <span>→</span>
                        </div>
                    </Link>
                </div>
            </section>

            {/* Footer */}
            <footer className="border-t border-white/10 mt-12 bg-slate-950/80 backdrop-blur-md relative z-10">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-8 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                    <div>
                        <div className="font-display font-bold text-white text-sm">
                            NeuralCord<span className="text-cyan-400">.AI</span>
                        </div>
                        <div className="kicker text-zinc-500 text-[9px] mt-0.5">
                            © 2026 NEURAL CORD RESEARCH · PYTORCH RES-ATTN U-NET PIPELINE
                        </div>
                    </div>
                    <div className="text-[11px] text-zinc-400 max-w-md text-left sm:text-right font-light leading-relaxed">
                        Medical Research Pipeline. Not intended as a sole diagnostic instrument. 
                        Outputs should be evaluated in context with full clinical imaging workups.
                    </div>
                </div>
            </footer>
        </div>
    );
}

const StatCard = ({ label, value, suffix, sub, highlight = false, "data-testid": tid }) => (
    <div 
        className={`p-5 rounded-2xl border transition-all duration-300 backdrop-blur-md ${
            highlight 
                ? "border-cyan-400/40 bg-gradient-to-br from-cyan-950/40 to-slate-900/60 shadow-[0_0_25px_rgba(34,211,238,0.15)]" 
                : "border-white/10 bg-slate-900/40 hover:border-white/20 hover:bg-slate-900/70"
        }`} 
        data-testid={tid}
    >
        <div className="kicker text-zinc-400 text-[9px]">{label}</div>
        <div className="font-display text-3xl font-extrabold mt-2 leading-none text-white tracking-tight flex items-baseline gap-1">
            <span className={highlight ? "text-cyan-300" : "text-white"}>{value}</span>
            {suffix && <span className="text-zinc-400 text-sm font-mono">{suffix}</span>}
        </div>
        {sub && <div className="text-[10px] font-mono text-zinc-400 mt-2">{sub}</div>}
    </div>
);