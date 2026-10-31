import Header from "../components/Header";
import TrainingGraphsPanel from "../components/TrainingGraphsPanel";

export default function TrainingCurvesPage() {
    return (
        <div className="min-h-screen bg-transparent relative overflow-x-clip" data-testid="training-curves-page">
            <Header />

            {/* Page Hero Header */}
            <section className="border-b border-white/10 relative z-10 fade-up bg-gradient-to-b from-slate-950/80 via-slate-900/30 to-transparent">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-purple-400/30 bg-purple-950/40 text-purple-300 font-mono text-xs mb-4 shadow-[0_0_20px_rgba(168,85,247,0.15)]">
                        <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></span>
                        <span>EPOCH DYNAMICS & OPTIMIZATION TRAJECTORY</span>
                    </div>
                    <h1 className="font-display text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
                        Training & Validation Epoch Curves
                    </h1>
                    <p className="mt-3 max-w-2xl text-zinc-300 text-base font-light leading-relaxed">
                        Track convergence dynamics across 30 training epochs. Inspect loss minimization, 
                        segmentation Dice score growth, and 3-model convergence comparisons.
                    </p>
                </div>
            </section>

            {/* Main Curves Content */}
            <div className="relative z-10">
                <TrainingGraphsPanel />
            </div>

            {/* Page Footer */}
            <footer className="border-t border-white/10 mt-16 bg-slate-950/80 backdrop-blur-md relative z-10">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-8 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                    <div>
                        <div className="font-display font-bold text-white text-sm">
                            NeuralCord<span className="text-cyan-400">.AI</span>
                        </div>
                        <div className="kicker text-zinc-500 text-[9px] mt-0.5">
                            © 2026 TRAINING EPOCH CONVERGENCE MONITOR
                        </div>
                    </div>
                    <div className="text-[11px] text-zinc-400 max-w-md text-left sm:text-right font-light leading-relaxed">
                        Optimized with Cosine Annealing Learning Rate Schedule and Combined Dice-Focal Loss.
                    </div>
                </div>
            </footer>
        </div>
    );
}
