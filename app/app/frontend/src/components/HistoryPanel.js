import { Trash2, History, ArrowUpRight, Clock } from "lucide-react";

export default function HistoryPanel({ items, onSelect, onDelete }) {
    return (
        <section id="history" className="border-t border-white/10 relative z-10 bg-slate-950/30" data-testid="history-section">
            <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12">
                <div className="flex items-end justify-between mb-6">
                    <div>
                        <div className="kicker text-cyan-400 flex items-center gap-2">
                            <History className="w-3.5 h-3.5" />
                            05 · SESSION AUDIT TRAIL
                        </div>
                        <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight mt-1 text-white">
                            Diagnostic History Log
                        </h2>
                    </div>
                    <span className="kicker text-zinc-400 font-mono text-xs px-3 py-1 rounded-full bg-white/5 border border-white/10">
                        {items.length} LOGGED RECORDS
                    </span>
                </div>

                {items.length === 0 ? (
                    <div className="glass-card p-12 text-center text-zinc-400 text-sm border border-white/10 rounded-2xl" data-testid="history-empty">
                        <Clock className="w-8 h-8 text-zinc-500 mx-auto mb-2 opacity-50" />
                        <div className="font-display text-white font-semibold">No predictions logged yet.</div>
                        <div className="text-xs text-zinc-500 mt-1">Upload an MRI slice or click any demo sample above to generate analysis records.</div>
                    </div>
                ) : (
                    <div className="glass-panel overflow-x-auto rounded-2xl border border-white/10 shadow-2xl">
                        <table className="w-full text-sm" data-testid="history-table">
                            <thead className="border-b border-white/10 bg-slate-950/70">
                                <tr className="text-left text-xs font-mono text-zinc-400">
                                    <Th>FILE IDENTIFIER</Th>
                                    <Th>PREDICTED CLASS</Th>
                                    <Th>CONFIDENCE</Th>
                                    <Th>U-NET DICE</Th>
                                    <Th>IoU</Th>
                                    <Th>LESIONS</Th>
                                    <Th>TIMESTAMP</Th>
                                    <Th></Th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                                {items.map((rec) => (
                                    <tr
                                        key={rec.id}
                                        className="hover:bg-cyan-500/[0.06] cursor-pointer transition-colors group"
                                        onClick={() => onSelect(rec)}
                                        data-testid={`history-row-${rec.id}`}
                                    >
                                        <Td className="font-mono text-zinc-200 max-w-[220px] truncate font-medium flex items-center gap-2">
                                            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 opacity-60 group-hover:opacity-100" />
                                            {rec.filename}
                                        </Td>

                                        <Td>
                                            <span className={`inline-block px-2.5 py-1 rounded-md border bg-cls-${rec.predicted_class} font-mono text-xs font-bold`}>
                                                {rec.predicted_class}
                                            </span>
                                        </Td>

                                        <Td className="font-mono text-white font-semibold">{rec.confidence.toFixed(1)}%</Td>
                                        <Td className="font-mono text-emerald-400 font-semibold">{rec.dice_score.toFixed(3)}</Td>
                                        <Td className="font-mono text-zinc-300">{rec.iou_score.toFixed(3)}</Td>
                                        <Td className="font-mono text-zinc-300">{rec.lesion_count}</Td>
                                        <Td className="text-zinc-400 text-xs font-mono">
                                            {new Date(rec.timestamp).toLocaleString()}
                                        </Td>

                                        <Td>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onDelete(rec.id);
                                                }}
                                                className="p-1.5 rounded-lg hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-colors"
                                                data-testid={`history-delete-${rec.id}`}
                                                title="Delete entry"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </Td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </section>
    );
}

const Th = ({ children }) => (
    <th className="kicker px-5 py-4 font-semibold text-zinc-400 tracking-wider text-[10px]">{children}</th>
);

const Td = ({ children, className = "" }) => (
    <td className={`px-5 py-3.5 ${className}`}>{children}</td>
);