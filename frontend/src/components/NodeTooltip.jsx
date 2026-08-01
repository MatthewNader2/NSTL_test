import { useState, useEffect } from "react";
import { useStore } from "../store";

export default function NodeTooltip() {
  const hoveredNode = useStore((s) => s.hoveredNode);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handler = (e) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", handler);
    return () => window.removeEventListener("mousemove", handler);
  }, []);

  if (!hoveredNode) return null;

  return (
    <div
      className="node-tooltip"
      style={{ left: pos.x + 18, top: pos.y - 14 }}
    >
      <div style={{ fontWeight: 700, color: "var(--cyan)", fontSize: "0.88rem", marginBottom: 4 }}>
        {hoveredNode.cell_id}
      </div>
      <div style={{ color: "var(--t2)", fontSize: "0.72rem", display: "flex", gap: 8 }}>
        <span>Stage {hoveredNode.stage}</span>
        <span style={{ opacity: 0.4 }}>·</span>
        <span>{hoveredNode.type?.toUpperCase()}</span>
      </div>
      {hoveredNode.keywords?.slice(0, 3).map((k) => (
        <span key={k} style={{ fontSize: "0.62rem", color: "var(--t3)", marginRight: 4 }}>#{k}</span>
      ))}
    </div>
  );
}
