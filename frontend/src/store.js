import { create } from "zustand";

export const useStore = create((set) => ({
  cells: [],
  setCells: (cells) => set({ cells }),
  activePath: [],
  setActivePath: (path) => set({ activePath: path }),
  virtualEdges: new Set(),
  setVirtualEdges: (edges) => set({ virtualEdges: edges }),
  generatedCode: "# Generated code will appear here\n",
  setGeneratedCode: (code) => set({ generatedCode: code }),
  logs: [],
  addLog: (log) => set((state) => ({ logs: [...state.logs.slice(-200), log] })),
  selectedNode: null,
  setSelectedNode: (node) => set({ selectedNode: node }),
  hoveredNode: null,
  setHoveredNode: (node) => set({ hoveredNode: node }),
  apiStatus: "connecting",
  setApiStatus: (status) => set({ apiStatus: status }),
  rightActiveTab: "code", // new: which tab is open in right panel
  setRightActiveTab: (tab) => set({ rightActiveTab: tab }),
}));
