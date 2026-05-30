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
      style={{
        position: "fixed",
        left: pos.x + 20,
        top: pos.y - 20,
        background: "rgba(10, 15, 30, 0.95)",
        border: "1px solid rgba(0, 229, 255, 0.4)",
        borderRadius: "8px",
        padding: "10px 14px",
        color: "var(--text-primary)",
        pointerEvents: "none",
        zIndex: 1000,
        boxShadow: "0 8px 24px rgba(0,0,0,0.8)",
        backdropFilter: "blur(10px)",
        transform: "translateY(0)",
        animation: "fadeIn 0.2s ease-out forwards",
      }}
    >
      <div
        style={{
          fontWeight: "bold",
          color: "var(--accent)",
          fontSize: "0.95rem",
          marginBottom: "4px",
        }}
      >
        {hoveredNode.cell_id}
      </div>
      <div
        style={{
          color: "var(--text-secondary)",
          fontSize: "0.8rem",
          display: "flex",
          gap: "8px",
        }}
      >
        <span>Stage {hoveredNode.stage}</span>
        <span>•</span>
        <span>{hoveredNode.type.toUpperCase()}</span>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(5px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
