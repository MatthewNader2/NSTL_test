import { useState, useEffect } from "react";
import Editor from "@monaco-editor/react";
import { useStore } from "../store";

export default function MonacoEditor() {
  const generatedCode = useStore((s) => s.generatedCode);
  const [editorFontSize, setEditorFontSize] = useState(12);

  useEffect(() => {
    const updateSize = () => {
      const rootFs = parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
      setEditorFontSize(Math.max(10, Math.min(24, rootFs * 0.75)));
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  return (
    <div style={{ width: "100%", height: "100%", paddingTop: "6px", boxSizing: "border-box" }}>
      <Editor
        defaultLanguage="python"
        language="python"
        value={generatedCode}
        theme="vs-dark"
        options={{
          readOnly: true,
          minimap: { enabled: true, scale: 1, showSlider: "mouseover" },
          fontSize: editorFontSize,
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          wordWrap: "on",
          automaticLayout: true,
          tabSize: 4,
          renderLineHighlight: "none",
          overviewRulerBorder: false,
          hideCursorInOverviewRuler: true,
          smoothScrolling: true,
          cursorBlinking: "smooth",
          padding: { top: 12, bottom: 12 },
        }}
        loading={
          <div
            style={{
              color: "var(--text-secondary)",
              padding: 20,
              textAlign: "center",
            }}
          >
            Initializing code editor...
          </div>
        }
      />
    </div>
  );
}
