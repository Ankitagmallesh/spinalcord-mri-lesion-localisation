import { Eye, Sparkles, Upload, Activity, ShieldCheck, Flame, Layers } from "lucide-react";

export default function EmptyState({ onSampleSelect }) {
    return (
        <div className="glass-panel flex h-full min-h-[500px] flex-col overflow-hidden shadow-[0_12px_45px_rgba(0,0,0,0.5)]" data-testid="empty-state">
            <div className="flex items-center justify-between border-b border-white/10 bg-slate-950/60 px-5 py-4">
                <div className="kicker text-white font-semibold flex items-center gap-2">
                    <span className="flex h-2 w-2 rounded-full bg-cyan-400"></span>
                    VIEWER STANDBY · READY FOR MRI INFERENCE
                </div>
                <div className="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-3 py-1 text-[10px] font-mono text-cyan-300">
                    SYS · CALIBRATED
                </div>
            </div>

            <div className="relative flex-1 bg-[radial-gradient(circle_at_center,rgba(56,189,248,0.12),transparent_65%)] p-6 flex flex-col justify-between">
                
                {/* 4 Quadrant Reticle Frame */}
                <div className="grid grid-cols-2 gap-2 max-w-xl mx-auto w-full my-auto">
                    {[
                        { title: "T1 · Original Slice", sub: "CLAHE Preprocessed", icon: <Eye className="w-4 h-4 text-cyan-400" /> },
                        { title: "T2 · U-Net Cord", sub: "Neural Boundary Mask", icon: <Layers className="w-4 h-4 text-blue-400" /> },
                        { title: "T3 · Lesion Map", sub: "Intramedullary Boxes", icon: <Activity className="w-4 h-4 text-amber-400" /> },
                        { title: "T4 · Grad-CAM XAI", sub: "Decision Saliency", icon: <Flame className="w-4 h-4 text-red-400" /> },
                    ].map((item, i) => (
                        <div key={i} className="relative aspect-video rounded-xl bg-slate-950/70 border border-white/10 p-4 flex flex-col justify-between overflow-hidden group hover:border-cyan-400/40 transition-colors">
                            <div className="hud-corner-tl" />
                            <div className="hud-corner-tr" />
                            <div className="hud-corner-bl" />
                            <div className="hud-corner-br" />
                            
                            <div className="flex items-center justify-between">
                                <span className="font-mono text-xs font-bold text-zinc-300">{item.title}</span>
                                {item.icon}
                            </div>
                            <div className="text-[11px] font-mono text-zinc-400">{item.sub}</div>
                        </div>
                    ))}
                </div>

                {/* Bottom Guide Prompt */}
                <div className="mt-6 rounded-xl border border-white/10 bg-slate-950/80 p-4 backdrop-blur-md max-w-xl mx-auto w-full text-center">
                    <div className="kicker text-cyan-400 text-xs">READY FOR SPINAL CORD ANALYSIS</div>
                    <div className="font-display text-base font-bold text-white mt-1">
                        Upload an MRI slice on the left or click any demo sample to launch inference.
                    </div>
                    <div className="mt-2 text-xs text-zinc-400 font-light">
                        Dual-task neural model generates cord segmentation, lesion bounding boxes, and diagnostic pathology predictions.
                    </div>
                </div>
            </div>

            <div className="flex items-center justify-between border-t border-white/10 bg-slate-950/60 px-5 py-3.5">
                <span className="kicker text-zinc-300 text-[10px]">AWAITING TENSOR INFERENCE</span>
                <span className="kicker text-cyan-400 text-[10px] font-mono">AUTOMATIC RESIZE 256×256</span>
            </div>
        </div>
    );
}

