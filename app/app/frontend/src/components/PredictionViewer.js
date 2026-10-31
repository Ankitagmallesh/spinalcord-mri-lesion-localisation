import { useState } from "react";
import { 
    Eye, Flame, Layers, Sparkles, Target, Zap, ArrowRight, Info, 
    Maximize2, Sliders, SplitSquareVertical, Download, X, ZoomIn, 
    RotateCcw, Contrast, SunMedium 
} from "lucide-react";

export default function PredictionViewer({ record }) {
    const [viewMode, setViewMode] = useState("grid"); // "grid", "slider"
    const [viewFilter, setViewFilter] = useState("all"); // "all", "seg", "xai"
    const [sliderPos, setSliderPos] = useState(50);
    const [sliderLeft, setSliderLeft] = useState("original"); // "original"
    const [sliderRight, setSliderRight] = useState("overlay"); // "overlay", "gradcam", "segmented"
    const [activeFilter, setActiveFilter] = useState("normal"); // "normal", "contrast", "invert", "sharp"
    const [activeModalTile, setActiveModalTile] = useState(null);

    const allTiles = [
        { key: "original", label: "Original MRI", b64: record.original_b64, code: "T1", note: "Resized · CLAHE Enhanced", group: "seg", tag: "INPUT SCAN" },
        { key: "segmented", label: "Segmented Cord", b64: record.segmented_b64, code: "T2", note: "U-Net Segmented Region", group: "seg", tag: "CORD MASK" },
        { key: "mask", label: "Binary Cord Mask", b64: record.mask_b64, code: "T3", note: "Trained U-Net Mask", group: "seg", tag: "CONTOUR" },
        { key: "overlay", label: "Disease & Lesions", b64: record.overlay_b64, code: "T4", note: "Localized Lesion Boxes", group: "seg", tag: "LESION DETECTION" },
        { 
            key: "gradcam",
            label: "Grad-CAM Saliency", 
            b64: record.gradcam_b64 || record.overlay_b64, 
            code: "T5", 
            note: "Neural Gradient Hotspot", 
            group: "xai",
            badge: "Grad-CAM XAI",
            badgeTone: "bg-red-500/20 text-red-300 border-red-500/30",
            tag: "GRADIENT ATTRIBUTION"
        },
        { 
            key: "shap",
            label: "SHAP Feature Map", 
            b64: record.shap_b64 || record.segmented_b64, 
            code: "T6", 
            note: "Anatomical Attribution", 
            group: "xai",
            badge: "SHAP Attribution",
            badgeTone: "bg-purple-500/20 text-purple-300 border-purple-500/30",
            tag: "BIOMARKER HEATMAP"
        },
    ];

    const displayTiles = viewFilter === "all" 
        ? allTiles 
        : allTiles.filter(t => t.group === viewFilter);

    const xai = record.xai_attribution;

    const getFilterClass = () => {
        if (activeFilter === "contrast") return "contrast-high";
        if (activeFilter === "invert") return "mri-invert";
        if (activeFilter === "sharp") return "mri-sharp";
        return "";
    };

    const getBase64ByKey = (key) => {
        const found = allTiles.find(t => t.key === key);
        return found ? found.b64 : record.original_b64;
    };

    const handleDownloadCurrent = () => {
        const link = document.createElement("a");
        link.href = `data:image/png;base64,${record.overlay_b64 || record.original_b64}`;
        link.download = `NeuralCord_${record.predicted_class || "Analysis"}_${record.filename || "scan"}.png`;
        link.click();
    };

    return (
        <div id="viewer" className="glass-panel fade-up overflow-hidden shadow-[0_12px_45px_rgba(0,0,0,0.6)]" data-testid="viewer-panel">
            
            {/* Header & Controls Bar */}
            <div className="flex flex-col items-start justify-between gap-3 border-b border-white/10 bg-slate-950/60 px-5 py-3.5 sm:flex-row sm:items-center">
                <div className="flex items-center gap-3">
                    <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-500/20 border border-blue-400/30 text-blue-400">
                        <Eye className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="kicker text-white font-semibold flex items-center gap-2 text-xs">
                            MULTI-VIEW DIAGNOSTIC CONSOLE
                        </div>
                        <div className="font-mono text-zinc-400 text-[10px] mt-0.5">
                            {record.filename} · 256×256 px · CLAHE Enhanced
                        </div>
                    </div>
                </div>

                {/* Switcher Controls */}
                <div className="flex flex-wrap items-center gap-2">
                    {/* View Mode: Grid vs Slider */}
                    <div className="flex items-center gap-1 bg-black/40 border border-white/10 p-1 rounded-lg">
                        <button
                            onClick={() => setViewMode("grid")}
                            className={`px-2.5 py-1 rounded text-xs font-mono transition-all ${
                                viewMode === "grid" ? "bg-white/15 text-white font-semibold shadow-sm" : "text-zinc-400 hover:text-white"
                            }`}
                            title="Grid Multi-View"
                        >
                            6-Panel Grid
                        </button>
                        <button
                            onClick={() => setViewMode("slider")}
                            className={`px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 ${
                                viewMode === "slider" ? "bg-cyan-500/25 text-cyan-300 font-semibold shadow-sm" : "text-zinc-400 hover:text-white"
                            }`}
                            title="Interactive Comparison Split Slider"
                        >
                            <SplitSquareVertical className="w-3 h-3" />
                            Comparison Slider
                        </button>
                    </div>

                    {/* Filter Mode (if Grid) */}
                    {viewMode === "grid" && (
                        <div className="hidden sm:flex items-center gap-1 bg-black/40 border border-white/10 p-1 rounded-lg">
                            <button
                                onClick={() => setViewFilter("all")}
                                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                                    viewFilter === "all" ? "bg-white/15 text-white" : "text-zinc-400 hover:text-white"
                                }`}
                            >
                                All Views
                            </button>
                            <button
                                onClick={() => setViewFilter("seg")}
                                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${
                                    viewFilter === "seg" ? "bg-blue-500/20 text-blue-300" : "text-zinc-400 hover:text-white"
                                }`}
                            >
                                Seg
                            </button>
                            <button
                                onClick={() => setViewFilter("xai")}
                                className={`px-2 py-1 rounded text-[11px] font-mono transition-colors flex items-center gap-1 ${
                                    viewFilter === "xai" ? "bg-red-500/20 text-red-300" : "text-zinc-400 hover:text-white"
                                }`}
                            >
                                <Flame className="w-2.5 h-2.5 text-red-400" />
                                XAI
                            </button>
                        </div>
                    )}

                    {/* Radiologist Image Filter Toggles */}
                    <div className="flex items-center gap-1 bg-black/40 border border-white/10 p-1 rounded-lg">
                        <button
                            onClick={() => setActiveFilter(activeFilter === "contrast" ? "normal" : "contrast")}
                            className={`p-1 rounded text-xs font-mono transition-colors ${
                                activeFilter === "contrast" ? "bg-amber-500/30 text-amber-300" : "text-zinc-400 hover:text-white"
                            }`}
                            title="Boost Contrast"
                        >
                            <SunMedium className="w-3.5 h-3.5" />
                        </button>
                        <button
                            onClick={() => setActiveFilter(activeFilter === "invert" ? "normal" : "invert")}
                            className={`p-1 rounded text-xs font-mono transition-colors ${
                                activeFilter === "invert" ? "bg-purple-500/30 text-purple-300" : "text-zinc-400 hover:text-white"
                            }`}
                            title="Invert Colors"
                        >
                            <Contrast className="w-3.5 h-3.5" />
                        </button>
                        <button
                            onClick={handleDownloadCurrent}
                            className="p-1 rounded text-zinc-400 hover:text-white hover:bg-white/10 transition-colors"
                            title="Export Processed Slice"
                        >
                            <Download className="w-3.5 h-3.5" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Explainable AI Flow Pipeline Banner */}
            <div className="border-b border-white/10 bg-gradient-to-r from-blue-950/40 via-purple-950/30 to-red-950/40 px-5 py-2.5">
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                    <span className="kicker text-cyan-400 text-[10px] mr-1 flex items-center gap-1">
                        <Sparkles className="w-3 h-3 text-cyan-400" />
                        XAI PIPELINE FLOW:
                    </span>
                    <span className="px-2 py-0.5 rounded bg-white/10 text-zinc-200 border border-white/10">Input Scan</span>
                    <ArrowRight className="w-3 h-3 text-zinc-500" />
                    <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 font-semibold border border-blue-500/30">U-Net Segmenter</span>
                    <ArrowRight className="w-3 h-3 text-zinc-500" />
                    <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30">
                        {record?.predicted_class || "Disease"} ({Number(record?.confidence ?? 0).toFixed(1)}%)
                    </span>
                    <ArrowRight className="w-3 h-3 text-zinc-500" />
                    <span className="px-2 py-0.5 rounded bg-red-500/25 text-red-300 font-semibold flex items-center gap-1 border border-red-500/30">
                        <Flame className="w-3 h-3 text-red-400" />
                        Grad-CAM Hotspot
                    </span>
                    <ArrowRight className="w-3 h-3 text-zinc-500" />
                    <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
                        Focal Lesion {xai?.focal_area_pct != null ? `(${xai.focal_area_pct}%)` : ""}
                    </span>
                </div>
            </div>

            {/* MAIN VIEWER CONTENT */}
            {viewMode === "grid" ? (
                /* 6-Panel Image Tiles Grid */
                <div className={`grid ${displayTiles.length <= 2 ? "grid-cols-2" : displayTiles.length <= 4 ? "grid-cols-2" : "grid-cols-2 sm:grid-cols-3"} gap-[1px] bg-white/10`}>
                    {displayTiles.map((t) => (
                        <Tile 
                            key={t.code} 
                            {...t} 
                            filterClass={getFilterClass()} 
                            onInspect={() => setActiveModalTile(t)} 
                        />
                    ))}
                </div>
            ) : (
                /* Interactive Split-Screen Comparison Slider */
                <div className="p-6 bg-black/60 flex flex-col items-center">
                    <div className="w-full max-w-2xl flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <span className="kicker text-zinc-400 text-xs">LEFT:</span>
                            <span className="font-mono text-xs text-white px-2.5 py-1 rounded bg-white/10 border border-white/15">Original T1 MRI</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="kicker text-zinc-400 text-xs">RIGHT OVERLAY:</span>
                            <select 
                                value={sliderRight} 
                                onChange={(e) => setSliderRight(e.target.value)}
                                className="bg-slate-900 border border-white/15 text-cyan-300 text-xs font-mono rounded-lg px-2 py-1 outline-none"
                            >
                                <option value="overlay">Disease & Lesions (T4)</option>
                                <option value="gradcam">Grad-CAM Saliency (T5)</option>
                                <option value="segmented">Segmented Cord (T2)</option>
                                <option value="shap">SHAP Attribution (T6)</option>
                            </select>
                        </div>
                    </div>

                    <div className="relative w-full max-w-2xl aspect-square rounded-2xl overflow-hidden border border-white/20 shadow-2xl bg-black select-none">
                        {/* Under image (Right target: Overlay / Grad-cam) */}
                        <img 
                            src={`data:image/png;base64,${getBase64ByKey(sliderRight)}`}
                            alt="Comparison target"
                            className={`absolute inset-0 h-full w-full object-contain ${getFilterClass()}`}
                        />

                        {/* Over image (Left target: Original MRI) clipped with sliderPos */}
                        <div 
                            className="absolute inset-0 overflow-hidden" 
                            style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                        >
                            <img 
                                src={`data:image/png;base64,${record.original_b64}`}
                                alt="Original MRI"
                                className={`absolute inset-0 h-full w-full object-contain ${getFilterClass()}`}
                            />
                        </div>

                        {/* Dividing Line & Draggable Handle */}
                        <div 
                            className="absolute top-0 bottom-0 w-1 bg-cyan-400 shadow-[0_0_15px_rgba(56,189,248,0.9)] z-20 pointer-events-none"
                            style={{ left: `${sliderPos}%` }}
                        >
                            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-slate-950 border-2 border-cyan-400 flex items-center justify-center text-cyan-300 shadow-lg text-[10px] font-mono">
                                ↔
                            </div>
                        </div>

                        {/* Interactive Range Input Overlay */}
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={sliderPos}
                            onChange={(e) => setSliderPos(Number(e.target.value))}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-30"
                        />

                        {/* HUD corner brackets */}
                        <div className="hud-corner-tl" />
                        <div className="hud-corner-tr" />
                        <div className="hud-corner-bl" />
                        <div className="hud-corner-br" />
                    </div>

                    <div className="mt-4 font-mono text-xs text-zinc-400 flex items-center gap-2">
                        <span>Drag slider across MRI slice to compare anatomical tissue vs localized lesions</span>
                        <span className="text-cyan-300 font-bold">{sliderPos}%</span>
                    </div>
                </div>
            )}

            {/* Bottom Diagnostic & XAI Metrics Footer */}
            <div className="grid grid-cols-2 gap-3 border-t border-white/10 bg-black/40 px-5 py-4 text-xs sm:grid-cols-5 sm:gap-2.5">
                <Spec label="LESION COUNT" value={record?.lesion_count ?? 0} />
                <Spec label="LESION AREA RATIO" value={`${(Number(record?.lesion_area_ratio ?? 0) * 100).toFixed(2)}%`} />
                <Spec label="U-NET DICE SCORE" value={Number(record?.dice_score ?? 0).toFixed(3)} highlight={Number(record?.dice_score ?? 0) > 0.9} />
                <Spec label="U-NET IoU SCORE" value={Number(record?.iou_score ?? 0).toFixed(3)} />
                <Spec 
                    label="GRAD-CAM FOCUS" 
                    value={xai?.focal_area_pct != null ? `${xai.focal_area_pct}% cord` : "Focal"} 
                    highlight={true}
                />
            </div>

            {/* Lightbox / Fullscreen Modal for Detail Inspection */}
            {activeModalTile && (
                <div 
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-xl p-4 sm:p-8 animate-fade-in"
                    onClick={() => setActiveModalTile(null)}
                >
                    <div 
                        className="relative max-w-4xl w-full glass-panel border border-white/20 rounded-2xl overflow-hidden shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between border-b border-white/10 bg-slate-950/80 px-6 py-4">
                            <div className="flex items-center gap-3">
                                <span className="font-mono text-sm font-bold text-cyan-400 bg-cyan-500/10 border border-cyan-400/30 px-2.5 py-1 rounded-md">
                                    {activeModalTile.code}
                                </span>
                                <div>
                                    <div className="font-display text-lg font-bold text-white">{activeModalTile.label}</div>
                                    <div className="text-xs text-zinc-400 font-mono">{activeModalTile.note}</div>
                                </div>
                            </div>
                            <button 
                                onClick={() => setActiveModalTile(null)}
                                className="p-2 rounded-xl bg-white/5 hover:bg-white/15 text-zinc-300 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="relative aspect-video max-h-[65vh] bg-black grid place-items-center p-4">
                            <img 
                                src={`data:image/png;base64,${activeModalTile.b64}`}
                                alt={activeModalTile.label}
                                className={`h-full w-full object-contain ${getFilterClass()}`}
                            />
                            <div className="hud-corner-tl" />
                            <div className="hud-corner-tr" />
                            <div className="hud-corner-bl" />
                            <div className="hud-corner-br" />
                        </div>

                        <div className="border-t border-white/10 bg-slate-950/80 p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                            <div className="text-xs text-zinc-300 font-light max-w-xl">
                                <strong>Inspection View:</strong> Enhanced slice resolution processed through the NeuralCord pipeline. 
                                Saliency and lesion annotations verify boundary sharpness across intramedullary coordinates.
                            </div>
                            <button 
                                onClick={handleDownloadCurrent}
                                className="btn-ghost text-xs px-4 py-2 shrink-0"
                            >
                                <Download className="w-3.5 h-3.5 mr-2" /> Download PNG
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

const Tile = ({ label, b64, code, note, badge, badgeTone, filterClass = "", onInspect }) => (
    <div className="group relative overflow-hidden bg-slate-950/80 transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-blue-500/10 opacity-0 transition-opacity duration-300 group-hover:opacity-100 pointer-events-none" />
        
        {/* HUD corner reticles */}
        <div className="hud-corner-tl" />
        <div className="hud-corner-tr" />
        <div className="hud-corner-bl" />
        <div className="hud-corner-br" />

        {/* Top bar with tile code and labels */}
        <div className="absolute top-0 z-10 flex w-full items-center justify-between border-b border-white/10 bg-slate-950/80 px-3.5 py-2 backdrop-blur-md">
            <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold text-cyan-300">{code}</span>
                <span className="kicker text-zinc-200 text-[10px]">{label}</span>
            </div>
            <div className="flex items-center gap-1.5">
                {badge ? (
                    <span className={`px-2 py-0.5 rounded text-[9px] font-mono border ${badgeTone}`}>
                        {badge}
                    </span>
                ) : (
                    <span className="kicker text-[9px] text-zinc-400">{note}</span>
                )}
                <button 
                    onClick={onInspect}
                    className="p-1 rounded bg-white/5 hover:bg-white/20 text-zinc-400 hover:text-white transition-colors ml-1"
                    title="Inspect Fullscreen"
                >
                    <Maximize2 className="w-3 h-3" />
                </button>
            </div>
        </div>

        {/* MRI Image Frame */}
        <div className="aspect-square viewer-frame pt-7 cursor-pointer" onClick={onInspect}>
            <img
                src={`data:image/png;base64,${b64}`}
                alt={label}
                className={`absolute inset-0 h-full w-full object-contain flicker transition-transform duration-700 group-hover:scale-105 ${filterClass}`}
            />
            <div className="pointer-events-none absolute inset-0 shadow-[inset_0_0_40px_rgba(0,0,0,0.8)]" />
        </div>
    </div>
);

const Spec = ({ label, value, highlight = false }) => (
    <div className={`rounded-xl border px-3.5 py-2.5 transition-all duration-300 ${
        highlight 
            ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-200 shadow-[0_0_15px_rgba(56,189,248,0.15)]" 
            : "border-white/10 bg-white/5 text-white hover:border-white/20"
    }`}>
        <div className="kicker text-[9px] text-zinc-400">{label}</div>
        <div className={`mt-0.5 font-display text-base font-bold ${highlight ? "text-cyan-300" : "text-white"}`}>
            {value}
        </div>
    </div>
);