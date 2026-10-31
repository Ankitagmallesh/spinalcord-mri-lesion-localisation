import { Award, Target, CheckCircle2, BarChart3, Activity } from "lucide-react";

export default function MetricsPanel({ metrics }) {
    if (!metrics) return null;
    const cm = metrics.confusion_matrix || [];
    const flatCm = Array.isArray(cm) ? cm.flat() : [];
    const max = flatCm.length > 0 ? Math.max(...flatCm, 1) : 1;
    const CLASSES = metrics.classes || ["Tumor", "MS", "Injury", "Normal"];
    const report = metrics.classification_report || {};

    return (
        <section id="metrics" className="border-t border-white/10 relative z-10 bg-slate-950/40" data-testid="metrics-section">
            <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-14 grid grid-cols-12 gap-8">
                
                <div className="col-span-12 lg:col-span-5">
                    <div className="kicker text-cyan-400 flex items-center gap-2">
                        <Activity className="w-3.5 h-3.5" />
                        03 · QUANTITATIVE MODEL EVALUATION
                    </div>
                    <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-tight mt-2 text-white">
                        Performance Metrics & Matrix
                    </h2>
                    <p className="text-zinc-300 text-sm mt-3 leading-relaxed max-w-md font-light">
                        Cross-validated aggregate diagnostic metrics over the benchmark dataset and active predictions.
                        Dice & IoU quantify segmentation fidelity; the confusion matrix validates disease classification.
                    </p>

                    <div className="grid grid-cols-2 gap-3 mt-6">
                        <StatCard label="Average Dice" value={Number(metrics.avg_dice ?? 0).toFixed(3)} sub="Cord Overlap" highlight={true} />
                        <StatCard label="Average IoU" value={Number(metrics.avg_iou ?? 0).toFixed(3)} sub="Jaccard Index" />
                        <StatCard label="Overall Accuracy" value={(Number(report?.accuracy ?? 0) * 100).toFixed(1) + "%"} sub="4-Class Diagnostic" />
                        <StatCard label="Class Count" value={String(CLASSES.length).padStart(2, '0')} sub="Pathology Categories" />
                    </div>

                    <div className="mt-6 border border-white/10 rounded-2xl overflow-hidden glass-card">
                        <div className="px-5 py-3.5 border-b border-white/10 kicker text-white bg-black/40 flex items-center justify-between">
                            <span>PER-CLASS F1 DIAGNOSTIC SCORE</span>
                            <span className="font-mono text-zinc-400 text-[10px]">WEIGHTED</span>
                        </div>
                        <div className="p-5 space-y-3">
                            {CLASSES.map((c) => {
                                const r = report[c] || { f1: 0 };
                                const f1Val = Number(r.f1 ?? 0);
                                return (
                                    <div key={c} className="grid grid-cols-12 items-center gap-2 text-xs">
                                        <span className={`col-span-3 font-mono font-semibold cls-${c}`}>{c}</span>
                                        <div className="col-span-7 h-2 progress-track relative bg-black/60 rounded-full">
                                            <div
                                                className={`progress-fill shadow-[0_0_10px_currentcolor] bg-cls-${c} text-cls-${c}`}
                                                style={{ width: `${Math.min(100, Math.max(0, f1Val * 100))}%` }}
                                            />
                                        </div>
                                        <span className="col-span-2 text-right font-mono text-white font-bold">
                                            {f1Val.toFixed(3)}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>

                <div className="col-span-12 lg:col-span-7">
                    <div className="border border-white/10 rounded-t-xl overflow-hidden glass-card">
                        <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between bg-black/20">
                            <span className="kicker text-white">CONFUSION MATRIX</span>
                            <span className="kicker text-zinc-500">PREDICTED →</span>
                        </div>

                        <div className="p-8">
                            <div className="grid" style={{ gridTemplateColumns: `100px repeat(${CLASSES.length}, 1fr)` }}>
                                <div />
                                {CLASSES.map((c) => (
                                    <div key={c} className={`text-center kicker cls-${c}`}>
                                        {c}
                                    </div>
                                ))}

                                {cm.map((row, i) => (
                                    <RowCells key={i} row={row} rowIndex={i} max={max} actual={CLASSES[i]} CLASSES={CLASSES} />
                                ))}
                            </div>

                            <div className="mt-6 kicker text-zinc-500">↑ ACTUAL</div>
                        </div>
                    </div>

                    <div 
                        className="grid gap-px bg-white/10 border border-white/10 border-t-0 rounded-b-xl overflow-hidden glass-panel"
                        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}
                    >
                        {CLASSES.map((c) => {
                            const r = (report && report[c]) || {};
                            return (
                                <div key={c} className="bg-black/40 p-4 transition-colors hover:bg-white/5" data-testid={`metric-card-${c}`}>
                                    <div className={`kicker cls-${c}`}>{c}</div>
                                    <div className="grid grid-cols-3 gap-2 mt-3 text-sm font-mono text-white">
                                        <div>
                                            <div className="text-zinc-500 text-[10px] mb-1">Acc</div>
                                            {Number((r.accuracy ?? r.recall) || 0).toFixed(2)}
                                        </div>
                                        <div>
                                            <div className="text-zinc-500 text-[10px] mb-1">Avg</div>
                                            {Number(r.avg_dice || 0).toFixed(2)}
                                        </div>
                                        <div>
                                            <div className="text-zinc-500 text-[10px] mb-1">IoU</div>
                                            {Number(r.avg_iou || 0).toFixed(2)}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </section>
    );
}

const RowCells = ({ row, rowIndex, max, actual, CLASSES }) => {
    const rowList = Array.isArray(row) ? row : [];
    const rowSum = rowList.reduce((a, b) => a + b, 0);
    return (
        <>
            <div className={`kicker cls-${actual} flex items-center pr-3 justify-end`}>
                {actual}
            </div>

            {row.map((v, j) => {
                const isCorrect = rowIndex === j;
                const intensity = rowSum > 0 ? v / rowSum : 0;
                
                const rgb = isCorrect ? '34, 197, 94' : '239, 68, 68';
                const predictedClass = CLASSES[j];
                const tooltip = `Actual: ${actual}\nPredicted: ${predictedClass}\nCount: ${v}`;

                return (
                    <div
                        key={j}
                        title={tooltip}
                        className="group relative aspect-square border border-white/5 flex flex-col items-center justify-center font-mono text-sm transition-all duration-300 hover:scale-110 hover:z-10 rounded-md m-0.5 cursor-help"
                        style={{
                            backgroundColor: v > 0 ? `rgba(${rgb}, ${Math.max(0.1, intensity)})` : 'rgba(255,255,255,0.02)',
                            color: intensity > 0.4 ? "white" : "rgba(255,255,255,0.5)",
                            boxShadow: intensity > 0.5 ? `0 0 20px rgba(${rgb}, ${intensity * 0.4})` : 'none',
                        }}
                    >
                        <span className="text-base font-semibold">{v}</span>
                    </div>
                );
            })}
        </>
    );
};

const StatCard = ({ label, value, sub, highlight = false }) => (
    <div className={`p-4 rounded-xl border transition-all duration-300 ${
        highlight 
            ? "border-cyan-400/40 bg-gradient-to-br from-cyan-950/40 to-slate-900/60 text-cyan-200 shadow-[0_0_20px_rgba(34,211,238,0.15)]" 
            : "border-white/10 bg-slate-900/50 text-white hover:border-white/20"
    }`}>
        <div className="kicker text-[9px] text-zinc-400">{label}</div>
        <div className={`font-display text-2xl font-bold mt-1.5 ${highlight ? "text-cyan-300" : "text-white"}`}>{value}</div>
        {sub && <div className="text-[10px] font-mono text-zinc-400 mt-1">{sub}</div>}
    </div>
);