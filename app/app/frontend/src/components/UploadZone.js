import { useRef, useState } from "react";
import { Upload, FileImage, Loader2, Sparkles, Wand2, Check, AlertCircle } from "lucide-react";

export default function UploadZone({ onFile, onSampleSelect, loading, classes = ["Tumor", "MS", "Injury", "Normal"] }) {
    const inputRef = useRef(null);
    const [drag, setDrag] = useState(false);
    const [name, setName] = useState("");
    const [actualClass, setActualClass] = useState("");

    const handle = (file) => {
        if (!file) return;
        setName(file.name);
        onFile(file, actualClass);
    };

    const handlePresetSample = async (clsName) => {
        if (loading) return;
        if (onSampleSelect) {
            setName(`Sample_${clsName}.png`);
            onSampleSelect(clsName);
        }
    };

    return (
        <div className="glass-card relative flex flex-col overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.45)]" data-testid="upload-card">
            {/* Top Bar */}
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-3.5 bg-black/40 backdrop-blur-md">
                <div className="kicker text-white font-semibold flex items-center gap-2">
                    <span className="flex h-2 w-2 rounded-full bg-cyan-400"></span>
                    01 · INPUT SCAN TERMINAL
                </div>
                <div className="flex items-center gap-1.5 font-mono text-[10px] text-zinc-400">
                    <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-cyan-300">PNG</span>
                    <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-cyan-300">JPG</span>
                </div>
            </div>

            {/* Drop Zone */}
            <div
                onDragOver={(e) => {
                    e.preventDefault();
                    setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                    e.preventDefault();
                    setDrag(false);
                    handle(e.dataTransfer.files?.[0]);
                }}
                onClick={() => !loading && inputRef.current?.click()}
                className={`relative m-4 cursor-pointer rounded-2xl border-2 border-dashed p-7 text-center transition-all duration-300 overflow-hidden ${
                    drag
                        ? "border-cyan-400 bg-cyan-500/15 shadow-[0_0_35px_rgba(34,211,238,0.25)] scale-[0.99]"
                        : "border-white/15 bg-gradient-to-b from-white/[0.04] to-transparent hover:border-cyan-400/50 hover:bg-white/[0.07]"
                }`}
                data-testid="upload-mri-zone"
            >
                {/* Laser scan line effect during loading */}
                {loading && <div className="scan-laser" />}

                <div className="absolute inset-0 rounded-2xl bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.15),transparent_65%)] pointer-events-none" />
                
                <input
                    ref={inputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    className="hidden"
                    onChange={(e) => handle(e.target.files?.[0])}
                    data-testid="upload-input"
                />

                <div className="relative flex flex-col items-center gap-3">
                    <div className="grid h-14 w-14 place-items-center rounded-2xl border border-white/15 bg-gradient-to-br from-white/10 to-white/5 shadow-lg backdrop-blur-md">
                        {loading ? (
                            <Loader2 className="h-7 w-7 animate-spin text-cyan-400" />
                        ) : (
                            <Upload className="h-6 w-6 text-cyan-300" />
                        )}
                    </div>

                    <div>
                        <div className="font-display text-lg font-bold tracking-tight text-white">
                            {loading ? "Analyzing Neural Slices..." : "Drop spinal MRI slice"}
                        </div>
                        <div className="mt-1 text-xs text-zinc-400 font-light max-w-xs leading-relaxed">
                            Click to browse or drag image. Automated CLAHE contrast & U-Net cord segmentation will execute.
                        </div>
                    </div>

                    {name && (
                        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-500/10 px-3.5 py-1.5 font-mono text-xs text-cyan-200 backdrop-blur-md">
                            <FileImage className="h-3.5 w-3.5 text-cyan-400" /> {name}
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Demo Sample Selector */}
            <div className="px-5 py-3 border-t border-b border-white/10 bg-black/20">
                <div className="flex items-center justify-between mb-2">
                    <span className="kicker text-[10px] text-zinc-400 flex items-center gap-1.5">
                        <Wand2 className="w-3 h-3 text-cyan-400" />
                        QUICK DEMO SAMPLES (1-CLICK TEST)
                    </span>
                </div>
                <div className="grid grid-cols-4 gap-1.5">
                    {classes.map((c) => (
                        <button
                            key={c}
                            type="button"
                            onClick={() => handlePresetSample(c)}
                            disabled={loading}
                            className={`px-2 py-1.5 rounded-lg text-xs font-mono font-medium transition-all duration-200 border text-center hover:scale-[1.02] active:scale-95 ${
                                c === "Tumor" ? "bg-red-500/10 hover:bg-red-500/20 text-red-300 border-red-500/30" :
                                c === "MS" ? "bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border-amber-500/30" :
                                c === "Injury" ? "bg-sky-500/10 hover:bg-sky-500/20 text-sky-300 border-sky-500/30" :
                                "bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                            }`}
                        >
                            {c}
                        </button>
                    ))}
                </div>
            </div>

            {/* Ground Truth Setting */}
            <div className="px-5 py-3 border-b border-white/10 flex items-center justify-between bg-black/10">
                <label className="text-xs text-zinc-400 font-mono">Ground Truth Label</label>
                <select 
                    value={actualClass}
                    onChange={(e) => setActualClass(e.target.value)}
                    className="bg-slate-900 border border-white/15 text-white text-xs rounded-lg px-2.5 py-1.5 outline-none focus:border-cyan-400 transition-colors font-mono"
                >
                    <option value="">Auto (From Filename)</option>
                    {classes.map(c => (
                        <option key={c} value={c}>{c}</option>
                    ))}
                </select>
            </div>

            {/* Pipeline Step Footnote */}
            <div className="mx-4 my-3 flex flex-col gap-1.5 rounded-xl border border-white/10 bg-black/30 px-3.5 py-2.5 text-xs text-zinc-400">
                <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-[11px] text-zinc-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                        Pipeline: CLAHE + U-Net + Grad-CAM
                    </span>
                    <span className="font-mono text-[10px] text-zinc-400">256×256 px</span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-emerald-400/90 border-t border-white/5 pt-1.5 font-mono">
                    <span className="flex items-center gap-1">
                        <Check className="w-3 h-3 text-emerald-400" />
                        Spinal Cord Verifier: Active
                    </span>
                    <span className="text-zinc-400">Rejects Non-Spine Scans</span>
                </div>
            </div>
        </div>
    );
}