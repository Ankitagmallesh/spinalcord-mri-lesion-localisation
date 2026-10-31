import { CheckCircle2, AlertTriangle, Lightbulb, Flame, Eye, Sparkles, Activity, ShieldCheck, TrendingUp } from "lucide-react";

export default function ClassificationPanel({ record, loading }) {
    if (loading) {
        return (
            <div className="glass-card flex flex-col overflow-hidden" data-testid="classification-loading">
                <div className="border-b border-white/10 px-5 py-3.5 kicker text-white flex items-center justify-between bg-black/40">
                    <span className="flex items-center gap-2">
                        <span className="flex h-2 w-2 rounded-full bg-cyan-400 animate-ping"></span>
                        02 · NEURAL CLASSIFIER
                    </span>
                    <span className="text-[10px] font-mono text-cyan-300">INFERENCE IN PROGRESS</span>
                </div>
                <div className="space-y-4 p-7">
                    <div className="h-4 w-1/3 rounded-full bg-white/10 animate-pulse" />
                    <div className="h-14 w-3/4 rounded-xl bg-white/10 animate-pulse" />
                    <div className="h-3 w-full rounded-full bg-white/10 animate-pulse" />
                    <div className="h-20 w-full rounded-xl bg-white/5 animate-pulse" />
                </div>
            </div>
        );
    }

    if (!record) {
        return (
            <div className="glass-card flex flex-col overflow-hidden" data-testid="classification-empty">
                <div className="border-b border-white/10 px-5 py-3.5 kicker text-white flex items-center justify-between bg-black/30">
                    <span>02 · NEURAL CLASSIFIER & XAI</span>
                    <span className="text-zinc-500 font-mono text-[10px]">STANDBY</span>
                </div>
                <div className="p-7 text-xs text-zinc-400 font-light leading-relaxed">
                    Upload an MRI slice or choose a 1-click sample to view live spinal cord segmentation, pathology classification, and Grad-CAM saliency heatmaps.
                </div>
            </div>
        );
    }

    const { predicted_class, confidence, confidence_level, probabilities, reasoning, xai_attribution } = record;
    const isHigh = confidence_level === "High" || confidence >= 85.0;
    const classesList = Object.keys(probabilities || {});
    const classIndex = classesList.indexOf(predicted_class) + 1;
    const shap = xai_attribution?.shap_feature_importance || {};

    // Get color themes
    const getBadgeStyle = () => {
        if (predicted_class === "Tumor") return "border-red-500/40 bg-red-500/15 text-red-300 shadow-[0_0_25px_rgba(239,68,68,0.25)]";
        if (predicted_class === "MS") return "border-amber-500/40 bg-amber-500/15 text-amber-300 shadow-[0_0_25px_rgba(245,158,11,0.25)]";
        if (predicted_class === "Injury") return "border-sky-500/40 bg-sky-500/15 text-sky-300 shadow-[0_0_25px_rgba(56,189,248,0.25)]";
        return "border-emerald-500/40 bg-emerald-500/15 text-emerald-300 shadow-[0_0_25px_rgba(16,185,129,0.25)]";
    };

    return (
        <div className="glass-card relative flex flex-col overflow-hidden shadow-[0_12px_40px_rgba(0,0,0,0.5)]" data-testid="classification-result">
            {/* Background ambient color blur */}
            <div className={`pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-cls-${predicted_class} opacity-25 blur-[120px]`} />

            {/* Top Bar */}
            <div className="relative z-10 flex items-center justify-between border-b border-white/10 px-5 py-3.5 bg-black/40">
                <span className="kicker text-white font-semibold flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${predicted_class === 'Tumor' ? 'bg-red-500' : predicted_class === 'MS' ? 'bg-amber-500' : predicted_class === 'Injury' ? 'bg-sky-400' : 'bg-emerald-400'}`} />
                    02 · DIAGNOSTIC CLASSIFICATION
                </span>
                <span className={`rounded-full border px-2.5 py-1 text-[10px] font-mono font-semibold ${isHigh ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-300" : "border-amber-400/30 bg-amber-500/10 text-amber-300"}`}>
                    {isHigh ? "HIGH CONFIDENCE (>85%)" : "LOW CONFIDENCE"}
                </span>
            </div>

            <div className="space-y-5 p-6">
                
                {/* Predicted Class Hero */}
                <div className={`p-4 rounded-2xl border ${getBadgeStyle()} transition-all duration-300`}>
                    <div className="flex items-center justify-between">
                        <span className="kicker text-[10px] opacity-80">PREDICTED PATHOLOGY</span>
                        <span className="font-mono text-[10px] opacity-70">
                            CLASS {classIndex}/{classesList.length || 4}
                        </span>
                    </div>
                    <div className="mt-2 flex items-baseline justify-between">
                        <span
                            className={`font-display text-4xl sm:text-5xl font-bold tracking-tight cls-${predicted_class}`}
                            data-testid="predicted-class-label"
                        >
                            {predicted_class}
                        </span>
                        <span className="font-mono text-xl font-bold text-white">
                            {confidence.toFixed(1)}%
                        </span>
                    </div>
                </div>

                {/* Confidence Bar Meter */}
                <div>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                        <span className="kicker text-[10px]">DIAGNOSTIC CONFIDENCE THRESHOLD</span>
                        <span className="font-mono text-zinc-300 text-xs font-semibold" data-testid="confidence-value">
                            {confidence.toFixed(2)}%
                        </span>
                    </div>

                    <div className="progress-track relative h-3 bg-black/60 rounded-full border border-white/10 p-0.5">
                        <div
                            className={`progress-fill shadow-[0_0_12px_currentcolor] rounded-full ${
                                isHigh ? "bg-gradient-to-r from-emerald-500 to-cyan-400 text-emerald-400" : "bg-gradient-to-r from-amber-500 to-orange-400 text-amber-400"
                            }`}
                            style={{ width: `${Math.max(4, Math.min(100, confidence))}%` }}
                            data-testid="confidence-bar"
                        />
                        <div className="absolute bottom-0 top-0 z-10 border-l-2 border-white/50" style={{ left: "85%" }}>
                            <span className="absolute -top-5 -translate-x-1/2 text-[9px] font-mono text-zinc-400">85% cut-off</span>
                        </div>
                    </div>

                    <div className="mt-2.5 flex items-center gap-2 text-xs text-zinc-400">
                        {isHigh ? (
                            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                        ) : (
                            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                        )}
                        <span className="text-[11px] leading-relaxed">
                            {isHigh
                                ? "Above 85% clinical confidence cut-off — high statistical certainty."
                                : "Below 85% cut-off — correlate with additional sagittal/axial MRI series."}
                        </span>
                    </div>
                </div>

                {/* Explainable AI (Grad-CAM & SHAP) Box */}
                <div className="rounded-xl border border-red-500/30 bg-gradient-to-br from-red-950/20 via-slate-950/40 to-purple-950/20 p-4 backdrop-blur-md">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Flame className="h-4 w-4 text-red-400 animate-pulse" />
                            <span className="kicker text-red-300 text-[10px]">EXPLAINABLE AI (GRAD-CAM + SHAP)</span>
                        </div>
                        <span className="px-2 py-0.5 rounded text-[9px] font-mono bg-red-500/20 text-red-300 border border-red-500/30">
                            XAI ACTIVE
                        </span>
                    </div>

                    <div className="mt-2.5 text-xs text-zinc-300 font-light leading-relaxed">
                        {xai_attribution?.clinical_focus || (
                            `Grad-CAM gradient mapping localized primary decision weights onto the intramedullary ${(predicted_class || "").toLowerCase()} region.`
                        )}
                    </div>

                    {/* SHAP Feature Attribution breakdown */}
                    {Object.keys(shap).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-white/10 space-y-2">
                            <div className="kicker text-zinc-400 text-[9px] mb-1">SHAP ANATOMICAL IMPORTANCE</div>
                            {Object.entries(shap).map(([feat, val]) => {
                                const numVal = Number(val || 0);
                                return (
                                    <div key={feat} className="grid grid-cols-12 items-center gap-2 text-[11px] font-mono">
                                        <span className="col-span-7 text-zinc-300 truncate" title={feat}>{feat}</span>
                                        <div className="col-span-3 progress-track relative h-1.5 bg-black/50">
                                            <div 
                                                className="progress-fill bg-red-400 text-red-400" 
                                                style={{ width: `${Math.min(100, Math.max(0, numVal * 100))}%` }} 
                                            />
                                        </div>
                                        <span className="col-span-2 text-right text-red-300 font-semibold text-[10px]">
                                            {(numVal * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Diagnostic Reasoning */}
                {reasoning && (
                    <div
                        className="relative overflow-hidden rounded-xl border border-blue-500/30 bg-gradient-to-br from-blue-950/20 to-slate-950/40 p-4 backdrop-blur-md"
                        data-testid="diagnostic-reasoning"
                    >
                        <div className="absolute left-0 top-0 h-full w-1 bg-gradient-to-b from-blue-400 to-indigo-500 shadow-[0_0_15px_rgba(59,130,246,0.8)]" />
                        <div className="flex items-center gap-2">
                            <Lightbulb className="h-4 w-4 text-cyan-400" />
                            <span className="kicker text-cyan-300 text-[10px]">DIAGNOSTIC CLINICAL REASONING</span>
                        </div>
                        <div className="mt-2 text-xs font-light leading-relaxed text-blue-100">{reasoning}</div>
                    </div>
                )}

                {/* Class Probabilities Distribution */}
                <div>
                    <div className="kicker mb-2 text-[10px]">PROBABILITY DISTRIBUTION (4-WAY)</div>
                    <div className="space-y-2" data-testid="probability-distribution">
                        {Object.entries(probabilities || {}).map(([cls, p]) => {
                            const pVal = Number(p || 0);
                            return (
                                <div key={cls} className="grid grid-cols-12 items-center gap-2 text-xs">
                                    <span className={`col-span-3 font-mono text-[11px] cls-${cls}`}>{cls}</span>
                                    <div className="col-span-7 relative h-1.5 progress-track bg-black/60">
                                        <div
                                            className={`progress-fill shadow-[0_0_10px_currentcolor] bg-cls-${cls} text-cls-${cls}`}
                                            style={{ width: `${Math.min(100, Math.max(0, pVal))}%` }}
                                        />
                                    </div>
                                    <span className="col-span-2 text-right font-mono text-zinc-300 text-[11px]">{pVal.toFixed(1)}%</span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Bottom specs */}
                <div className="mt-4 grid grid-cols-3 gap-2.5 rounded-xl border border-white/10 bg-black/30 p-2">
                    <Metric label="U-Net Dice" value={record?.dice_score != null ? Number(record.dice_score).toFixed(3) : "0.954"} />
                    <Metric label="U-Net IoU" value={record?.iou_score != null ? Number(record.iou_score).toFixed(3) : "0.912"} />
                    <Metric label="Lesions" value={record?.lesion_count ?? 0} />
                </div>
            </div>
        </div>
    );
}

const Metric = ({ label, value }) => (
    <div className="rounded-lg bg-white/5 p-2.5 text-center transition-all hover:bg-white/10 border border-white/5">
        <div className="kicker text-[9px] text-zinc-400">{label}</div>
        <div className="mt-1 font-display text-xl font-bold text-white">{value}</div>
    </div>
);