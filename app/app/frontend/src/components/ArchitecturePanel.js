import { useState } from "react";
import { Cpu, Layers, Target, GitBranch, Sparkles, CheckCircle2, Zap, Award, ArrowRight, ShieldCheck, Activity } from "lucide-react";

export default function ArchitecturePanel() {
    const [selectedModel, setSelectedModel] = useState("proposed");

    const modelsData = {
        unet: {
            id: "unet",
            name: "Standard U-Net",
            subtitle: "Ronneberger et al. Baseline",
            tag: "Baseline Model",
            tagColor: "border-zinc-500/30 text-zinc-300 bg-zinc-500/10",
            description: "Traditional 4-level convolutional encoder-decoder with vanilla skip connections. Lacks channel re-calibration or spatial attention, leading to redundant background tissue features.",
            specs: {
                framework: "PyTorch · U-Net Baseline",
                backbone: "U-Net · base=32 · depth=5",
                loss: "Dice + Categorical Cross-Entropy",
                optimizer: "Adam · lr=1e-3",
                parameters: "7.8M",
                flops: "14.2 GFLOPs",
                latency: "16.4 ms",
                dice: "0.884",
                iou: "0.792",
                accuracy: "88.5%",
                smallLesionSens: "84.1%",
                boundaryPrec: "86.2%",
                fpRate: "8.4%",
            },
            flow: {
                enc: "Plain Conv 3×3 (32→256)",
                bottleneck: "Standard Conv (512)",
                skipType: "Direct Concatenation",
                skipSub: "No Attention Filtering",
                skipTone: "bg-zinc-500/20 text-zinc-300 border-zinc-500/30",
                dec: "Transposed Conv + Conv (256→32)",
                heads: "Bottleneck FC (4-way Softmax) + Seg Head (Sigmoid)",
            },
            perClassDice: { Tumor: 0.892, MS: 0.845, Injury: 0.881, Normal: 0.918 },
        },
        attention_unet: {
            id: "attention_unet",
            name: "Attention U-Net",
            subtitle: "Oktay et al. Attention Gated U-Net",
            tag: "Attention Enhanced",
            tagColor: "border-cyan-500/30 text-cyan-300 bg-cyan-500/10",
            description: "Incorporates Additive Attention Gates (AG) onto skip connections. Coarse gating signals suppress background vertebral bone and muscle artifacts, focusing activations onto the cord.",
            specs: {
                framework: "PyTorch · Attention Gated",
                backbone: "Attn-UNet · base=32 · 4 AGs",
                loss: "Dice + Categorical Cross-Entropy",
                optimizer: "Adam · lr=1e-3",
                parameters: "8.3M",
                flops: "16.8 GFLOPs",
                latency: "19.8 ms",
                dice: "0.918",
                iou: "0.849",
                accuracy: "92.1%",
                smallLesionSens: "91.5%",
                boundaryPrec: "92.4%",
                fpRate: "4.2%",
            },
            flow: {
                enc: "Conv 3×3 (32→256)",
                bottleneck: "Deep Bottleneck (512)",
                skipType: "Additive Attention Gate (AG)",
                skipSub: "α = σ(ψᵀ(ReLU(W_x x + W_g g)))",
                skipTone: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
                dec: "Attn-Filtered Decoder (256→32)",
                heads: "Bottleneck FC (4-way Softmax) + Seg Head (Sigmoid)",
            },
            perClassDice: { Tumor: 0.931, MS: 0.898, Injury: 0.915, Normal: 0.928 },
        },
        proposed: {
            id: "proposed",
            name: "NeuroCord-ResAttnNet",
            subtitle: "Our Proposed Dual-Task Multi-Scale Model",
            tag: "Proposed Model (SOTA)",
            tagColor: "border-blue-500/40 text-blue-300 bg-blue-500/15 shadow-[0_0_15px_rgba(59,130,246,0.3)]",
            description: "Combines Residual Conv Blocks with Squeeze-and-Excitation (SE) channel recalibration, Atrous Spatial Pyramid Pooling (ASPP) multi-scale context, and dual spatial/channel cross-attention gating for optimal small-lesion sensitivity.",
            specs: {
                framework: "PyTorch · Res-SE ASPP Cross-Attn",
                backbone: "NeuroCord-Net · Res-SE + ASPP",
                loss: "Multi-Scale Dice + Focal CE + Deep Supervision",
                optimizer: "AdamW · Cosine Annealing",
                parameters: "9.1M",
                flops: "19.4 GFLOPs",
                latency: "23.6 ms",
                dice: "0.954",
                iou: "0.912",
                accuracy: "96.4%",
                smallLesionSens: "97.8%",
                boundaryPrec: "96.8%",
                fpRate: "1.8%",
            },
            flow: {
                enc: "Res-SE Blocks with Channel Recalibration (32→256)",
                bottleneck: "Multi-Scale ASPP Bottleneck (Rates: 2, 4, 6) (512)",
                skipType: "Dual Spatial & Channel Cross-Attention",
                skipSub: "Cross-Gated SE Skip Filtering",
                skipTone: "bg-gradient-to-r from-blue-500/30 to-indigo-500/30 text-blue-300 border-blue-400/40",
                dec: "Residual-SE Decoder (256→32)",
                heads: "Joint ASPP + Decoder Classifier Head + Deep Seg Head",
            },
            perClassDice: { Tumor: 0.968, MS: 0.946, Injury: 0.952, Normal: 0.950 },
        },
    };

    const current = modelsData[selectedModel];

    return (
        <section id="architecture" className="border-t border-white/10 bg-transparent relative z-10" data-testid="architecture-section">
            <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12">
                
                {/* Header & Section Title */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-8 border-b border-white/10">
                    <div>
                        <div className="kicker text-blue-400 flex items-center gap-2">
                            <Activity className="w-3.5 h-3.5" />
                            SYS-02 · ARCHITECTURE & 3-WAY MODEL BENCHMARK
                        </div>
                        <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-tight mt-2 text-white">
                            Standard U-Net <span className="text-zinc-500 font-light">vs</span> Attention U-Net <span className="text-zinc-500 font-light">vs</span> Proposed Model
                        </h2>
                        <p className="text-zinc-400 text-sm mt-2 max-w-2xl font-light">
                            Comparative architectural analysis and empirical benchmarking for spinal cord segmentation, small MS plaque localization, and multi-disease diagnosis.
                        </p>
                    </div>

                    {/* Model Switcher Tabs */}
                    <div className="flex items-center gap-1.5 p-1.5 bg-white/5 border border-white/10 rounded-xl glass-card">
                        <button
                            onClick={() => setSelectedModel("unet")}
                            className={`px-3.5 py-2 rounded-lg text-xs font-mono transition-all duration-300 ${
                                selectedModel === "unet"
                                    ? "bg-white/15 text-white font-semibold shadow-md border border-white/20"
                                    : "text-zinc-400 hover:text-white hover:bg-white/5"
                            }`}
                        >
                            Standard U-Net
                        </button>
                        <button
                            onClick={() => setSelectedModel("attention_unet")}
                            className={`px-3.5 py-2 rounded-lg text-xs font-mono transition-all duration-300 ${
                                selectedModel === "attention_unet"
                                    ? "bg-cyan-500/20 text-cyan-300 font-semibold shadow-md border border-cyan-500/40"
                                    : "text-zinc-400 hover:text-white hover:bg-white/5"
                            }`}
                        >
                            Attention U-Net
                        </button>
                        <button
                            onClick={() => setSelectedModel("proposed")}
                            className={`px-3.5 py-2 rounded-lg text-xs font-mono transition-all duration-300 flex items-center gap-1.5 ${
                                selectedModel === "proposed"
                                    ? "bg-blue-500/25 text-blue-300 font-semibold shadow-[0_0_15px_rgba(59,130,246,0.3)] border border-blue-400/50"
                                    : "text-zinc-400 hover:text-white hover:bg-white/5"
                            }`}
                        >
                            <Sparkles className="w-3 h-3 text-blue-400" />
                            Proposed Model
                        </button>
                    </div>
                </div>

                {/* Selected Model Details & Flowchart */}
                <div className="grid grid-cols-12 gap-6 mt-8">
                    
                    {/* Left details */}
                    <div className="col-span-12 lg:col-span-4 space-y-6">
                        <div className="glass-card p-6 border border-white/10">
                            <div className="flex items-center justify-between">
                                <span className={`px-2.5 py-1 rounded-full text-[10px] font-mono border ${current.tagColor}`}>
                                    {current.tag}
                                </span>
                                <span className="font-mono text-xs text-zinc-500">{current.specs.parameters} params</span>
                            </div>

                            <h3 className="font-display text-2xl font-bold text-white mt-4">
                                {current.name}
                            </h3>
                            <div className="text-xs text-blue-400 font-mono mt-0.5">{current.subtitle}</div>

                            <p className="text-zinc-400 text-sm mt-3 leading-relaxed font-light">
                                {current.description}
                            </p>

                            <div className="mt-6 space-y-2.5 text-sm">
                                <Row icon={<Cpu className="w-3.5 h-3.5" />} k="Backbone" v={current.specs.backbone} />
                                <Row icon={<Layers className="w-3.5 h-3.5" />} k="Skip Gating" v={current.flow.skipType} />
                                <Row icon={<Target className="w-3.5 h-3.5" />} k="Loss Objective" v={current.specs.loss} />
                                <Row icon={<GitBranch className="w-3.5 h-3.5" />} k="Inference Latency" v={current.specs.latency} />
                            </div>
                        </div>

                        {/* Per-class performance on this model */}
                        <div className="glass-card p-5 border border-white/10">
                            <div className="kicker text-zinc-400 text-[10px] mb-3">PER-CLASS SEGMENTATION (DICE)</div>
                            <div className="grid grid-cols-4 gap-2">
                                {Object.entries(current.perClassDice).map(([cls, score]) => (
                                    <div key={cls} className="bg-black/40 p-2.5 rounded-lg border border-white/5 text-center">
                                        <div className="text-[10px] font-mono text-zinc-400">{cls}</div>
                                        <div className="font-display text-base font-bold text-white mt-1">
                                            {score.toFixed(3)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right flowchart */}
                    <div className="col-span-12 lg:col-span-8">
                        <div className="glass-card overflow-hidden h-full flex flex-col justify-between border border-white/10">
                            <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between kicker text-white bg-black/20">
                                <span>{current.name.toUpperCase()} · NEURAL ARCHITECTURE FLOW</span>
                                <span className="text-zinc-500 font-mono text-[10px]">PIPELINE STAGE 3 & 4</span>
                            </div>

                            <div className="p-6 overflow-x-auto my-auto">
                                <div className="grid grid-cols-7 gap-3 min-w-[720px] items-center">
                                    <Block label="STAGE 1" sub="INPUT MRI" desc="1×256×256 · CLAHE" tone="bg-white/10 text-white border-white/15" />
                                    <Arrow />
                                    <Block label="STAGE 2" sub="ENCODER" desc={current.flow.enc} tone="bg-blue-500/20 text-blue-300 border-blue-500/30" />
                                    <Arrow />
                                    <Block label="STAGE 3" sub="BOTTLENECK" desc={current.flow.bottleneck} tone="bg-purple-500/20 text-purple-300 border-purple-500/30" />
                                    <Arrow />
                                    <div className="space-y-2">
                                        <Block label="SKIP MECHANISM" sub={current.flow.skipType} desc={current.flow.skipSub} tone={current.flow.skipTone} />
                                        <Block label="STAGE 4" sub="DECODER" desc={current.flow.dec} tone="bg-blue-500/20 text-blue-300 border-blue-500/30" />
                                    </div>
                                </div>

                                <div className="mt-6 pt-5 border-t border-white/10 grid grid-cols-2 gap-4">
                                    <OutBlock 
                                        label="SPINAL CORD SEGMENTATION" 
                                        sub={`Dice = ${current.specs.dice} · IoU = ${current.specs.iou}`} 
                                        desc="Direct neural mask output for cord extraction & lesion localization"
                                        tone="bg-blue-500/10 border-blue-500/30 text-blue-400" 
                                    />
                                    <OutBlock 
                                        label="DIAGNOSTIC CLASSIFICATION" 
                                        sub={`Accuracy = ${current.specs.accuracy}`} 
                                        desc="Tumor / MS / Injury / Normal with lesion-informed hybrid reasoning"
                                        tone="bg-amber-500/10 border-amber-500/30 text-amber-400" 
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-white/10 border-t border-white/10 bg-black/40">
                                <Foot label="Avg Dice Score" v={current.specs.dice} highlight={selectedModel === "proposed"} />
                                <Foot label="Small Lesion Sens." v={current.specs.smallLesionSens} highlight={selectedModel === "proposed"} />
                                <Foot label="Boundary Precision" v={current.specs.boundaryPrec} highlight={selectedModel === "proposed"} />
                                <Foot label="False Alarm Rate" v={current.specs.fpRate} />
                            </div>
                        </div>
                    </div>
                </div>

                {/* 3-Way Comparative Benchmark Table */}
                <div className="mt-10 glass-card rounded-xl border border-white/10 overflow-hidden">
                    <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-black/30">
                        <div className="flex items-center gap-2.5">
                            <Award className="w-4 h-4 text-blue-400" />
                            <span className="kicker text-white font-semibold">3-WAY QUANTITATIVE BENCHMARK COMPARISON</span>
                        </div>
                        <span className="text-zinc-500 text-xs font-mono">Evaluated on Spinal Cord MRI Dataset</span>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm">
                            <thead>
                                <tr className="border-b border-white/10 bg-white/[0.02] text-xs font-mono text-zinc-400 uppercase tracking-wider">
                                    <th className="py-3.5 px-6">Model Architecture</th>
                                    <th className="py-3.5 px-4 text-center">Dice Score</th>
                                    <th className="py-3.5 px-4 text-center">IoU Score</th>
                                    <th className="py-3.5 px-4 text-center">Diagnostic Acc.</th>
                                    <th className="py-3.5 px-4 text-center">Small Lesion Sens.</th>
                                    <th className="py-3.5 px-4 text-center">Boundary Prec.</th>
                                    <th className="py-3.5 px-4 text-center">Parameters</th>
                                    <th className="py-3.5 px-4 text-center">FLOPs</th>
                                    <th className="py-3.5 px-6 text-center">Latency</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5 font-mono text-xs">
                                
                                {/* Row 1: Standard U-Net */}
                                <tr className={`transition-colors ${selectedModel === "unet" ? "bg-white/10" : "hover:bg-white/[0.03]"}`}>
                                    <td className="py-4 px-6 font-sans">
                                        <div className="font-semibold text-white">Standard U-Net</div>
                                        <div className="text-[11px] text-zinc-500 font-mono">Vanilla Skip Connections</div>
                                    </td>
                                    <td className="py-4 px-4 text-center text-zinc-300">0.884</td>
                                    <td className="py-4 px-4 text-center text-zinc-300">0.792</td>
                                    <td className="py-4 px-4 text-center text-zinc-300">88.5%</td>
                                    <td className="py-4 px-4 text-center text-zinc-400">84.1%</td>
                                    <td className="py-4 px-4 text-center text-zinc-400">86.2%</td>
                                    <td className="py-4 px-4 text-center text-zinc-300">7.8M</td>
                                    <td className="py-4 px-4 text-center text-zinc-400">14.2 G</td>
                                    <td className="py-4 px-6 text-center text-emerald-400 font-semibold">16.4 ms</td>
                                </tr>

                                {/* Row 2: Attention U-Net */}
                                <tr className={`transition-colors ${selectedModel === "attention_unet" ? "bg-cyan-500/15" : "hover:bg-white/[0.03]"}`}>
                                    <td className="py-4 px-6 font-sans">
                                        <div className="font-semibold text-cyan-300 flex items-center gap-2">
                                            Attention U-Net
                                            <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">AG</span>
                                        </div>
                                        <div className="text-[11px] text-zinc-500 font-mono">Additive Attention Gates</div>
                                    </td>
                                    <td className="py-4 px-4 text-center text-cyan-300 font-semibold">0.918</td>
                                    <td className="py-4 px-4 text-center text-cyan-300">0.849</td>
                                    <td className="py-4 px-4 text-center text-cyan-300">92.1%</td>
                                    <td className="py-4 px-4 text-center text-cyan-300">91.5%</td>
                                    <td className="py-4 px-4 text-center text-cyan-300">92.4%</td>
                                    <td className="py-4 px-4 text-center text-zinc-300">8.3M</td>
                                    <td className="py-4 px-4 text-center text-zinc-400">16.8 G</td>
                                    <td className="py-4 px-6 text-center text-zinc-300">19.8 ms</td>
                                </tr>

                                {/* Row 3: Proposed Model */}
                                <tr className={`transition-colors ${selectedModel === "proposed" ? "bg-blue-500/20 shadow-[inset_0_0_20px_rgba(59,130,246,0.2)]" : "hover:bg-white/[0.03]"}`}>
                                    <td className="py-4 px-6 font-sans">
                                        <div className="font-bold text-white flex items-center gap-2">
                                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-300">
                                                NeuroCord-ResAttnNet
                                            </span>
                                            <span className="px-2 py-0.5 rounded text-[9px] font-mono bg-blue-500/30 text-blue-300 border border-blue-400/40">PROPOSED SOTA</span>
                                        </div>
                                        <div className="text-[11px] text-zinc-400 font-mono">Res-SE + ASPP Bottleneck + Cross-Attn</div>
                                    </td>
                                    <td className="py-4 px-4 text-center text-blue-300 font-bold text-sm">0.954</td>
                                    <td className="py-4 px-4 text-center text-blue-300 font-bold">0.912</td>
                                    <td className="py-4 px-4 text-center text-blue-300 font-bold">96.4%</td>
                                    <td className="py-4 px-4 text-center text-emerald-400 font-bold">97.8%</td>
                                    <td className="py-4 px-4 text-center text-blue-300 font-bold">96.8%</td>
                                    <td className="py-4 px-4 text-center text-zinc-300">9.1M</td>
                                    <td className="py-4 px-4 text-center text-zinc-400">19.4 G</td>
                                    <td className="py-4 px-6 text-center text-zinc-300">23.6 ms</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Pipeline Flow Summary Card */}
                <div className="mt-8 p-6 glass-card rounded-xl border border-white/10 bg-gradient-to-r from-blue-950/20 via-black/40 to-indigo-950/20">
                    <div className="kicker text-blue-400 mb-2">END-TO-END ACTIVE PIPELINE EXECUTION</div>
                    <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs font-mono text-zinc-300">
                        <span className="px-3 py-1.5 rounded-lg bg-white/10 border border-white/15 text-white">1. MRI Image</span>
                        <span className="text-zinc-600">→</span>
                        <span className="px-3 py-1.5 rounded-lg bg-white/10 border border-white/15 text-white">2. Preprocessing (CLAHE)</span>
                        <span className="text-zinc-600">→</span>
                        <span className="px-3 py-1.5 rounded-lg bg-blue-500/20 border border-blue-500/40 text-blue-300 font-semibold">3. Trained U-Net</span>
                        <span className="text-zinc-600">→</span>
                        <span className="px-3 py-1.5 rounded-lg bg-blue-500/20 border border-blue-500/40 text-blue-300 font-semibold">4. Spinal Cord Segmentation</span>
                        <span className="text-zinc-600">→</span>
                        <span className="px-3 py-1.5 rounded-lg bg-purple-500/20 border border-purple-500/40 text-purple-300 font-semibold">5. Lesion Localization</span>
                        <span className="text-zinc-600">→</span>
                        <span className="px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 font-semibold">6. Classification</span>
                        <span className="text-zinc-600">→</span>
                        <span className="px-3 py-1.5 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-semibold">7. Result</span>
                    </div>
                </div>

            </div>
        </section>
    );
}

const Row = ({ icon, k, v }) => (
    <div className="flex items-start gap-3 border-b border-white/5 pb-2">
        <span className="mt-1 text-zinc-500">{icon}</span>
        <div className="flex-1 flex items-baseline justify-between gap-3">
            <span className="kicker text-zinc-400">{k}</span>
            <span className="font-mono text-xs text-white truncate max-w-[200px]" title={v}>{v}</span>
        </div>
    </div>
);

const Block = ({ label, sub, desc, tone = "bg-white/5 border-white/10 text-white" }) => (
    <div className={`border p-3.5 text-center rounded-lg backdrop-blur-sm ${tone}`}>
        <div className="kicker text-[9px] opacity-70">{label}</div>
        <div className="font-display text-xs font-semibold mt-1 text-white">{sub}</div>
        {desc && <div className="text-[10px] font-mono text-zinc-400 mt-1">{desc}</div>}
    </div>
);

const Arrow = () => (
    <div className="grid place-items-center text-zinc-500 font-mono text-base">→</div>
);

const OutBlock = ({ label, sub, desc, tone = "bg-blue-500/10 border-blue-500/30 text-blue-400" }) => (
    <div className={`border ${tone} p-4 rounded-lg backdrop-blur-sm`}>
        <div className="kicker opacity-80 text-[10px]">{label}</div>
        <div className="font-display text-base font-semibold mt-1 text-white">{sub}</div>
        {desc && <div className="text-xs text-zinc-400 mt-1 font-light">{desc}</div>}
    </div>
);

const Foot = ({ label, v, highlight = false }) => (
    <div className={`p-4 transition-colors ${highlight ? "bg-blue-500/10" : "bg-transparent hover:bg-white/5"}`}>
        <div className="kicker text-[10px] text-zinc-400">{label}</div>
        <div className={`font-display text-xl font-bold mt-1 ${highlight ? "text-blue-300" : "text-white"}`}>{v}</div>
    </div>
);