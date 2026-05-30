import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as THREE from "three";
import {
  Terminal, Play, Copy, Download, Upload, Settings, Search,
  ChevronRight, ChevronDown, Cpu, Network, Zap, Activity,
  Box, X, Eye, Code2, Layers, GitBranch, AlertCircle,
  CheckCircle, Clock, Maximize2, RefreshCw, FileCode, Database,
  Crosshair, Info
} from "lucide-react";

// ─── MOCK DATA (mirrors what LatticeOrchestrator would load from trees/) ──────
const MOCK_CELLS = [
  // Stage 0 — Ingest
  { cell_id: "LOAD_CSV", stage: 0, type: "micro", keywords: ["load","csv","read","file","import"], inputs: { input_type: "str", expected_state: "source_identifier" }, outputs: { output_type: "dataframe", resulting_state: "raw_loaded" } },
  { cell_id: "LOAD_JSON", stage: 0, type: "micro", keywords: ["load","json","parse","file"], inputs: { input_type: "str", expected_state: "source_identifier" }, outputs: { output_type: "dataframe", resulting_state: "raw_loaded" } },
  { cell_id: "LOAD_EXCEL", stage: 0, type: "micro", keywords: ["load","excel","xlsx","spreadsheet"], inputs: { input_type: "str", expected_state: "source_identifier" }, outputs: { output_type: "dataframe", resulting_state: "raw_loaded" } },
  { cell_id: "LOAD_PARQUET", stage: 0, type: "micro", keywords: ["load","parquet","feather","fast"], inputs: { input_type: "str", expected_state: "source_identifier" }, outputs: { output_type: "dataframe", resulting_state: "raw_loaded" } },
  { cell_id: "STREAM_API", stage: 0, type: "micro", keywords: ["api","stream","endpoint","fetch","http"], inputs: { input_type: "str", expected_state: "source_identifier" }, outputs: { output_type: "dataframe", resulting_state: "raw_loaded" } },
  // Stage 1 — Clean
  { cell_id: "DROP_NULLS", stage: 1, type: "micro", keywords: ["clean","null","missing","drop","na"], inputs: { input_type: "dataframe", expected_state: "raw_loaded" }, outputs: { output_type: "dataframe", resulting_state: "null_cleaned" } },
  { cell_id: "FILL_NULLS", stage: 1, type: "micro", keywords: ["fill","impute","missing","forward","backward"], inputs: { input_type: "dataframe", expected_state: "raw_loaded" }, outputs: { output_type: "dataframe", resulting_state: "null_cleaned" } },
  { cell_id: "DEDUPLICATE", stage: 1, type: "micro", keywords: ["dedup","duplicate","unique","distinct"], inputs: { input_type: "dataframe", expected_state: "raw_loaded" }, outputs: { output_type: "dataframe", resulting_state: "deduped" } },
  { cell_id: "TYPE_CAST", stage: 1, type: "micro", keywords: ["cast","type","convert","dtype","schema"], inputs: { input_type: "dataframe", expected_state: "null_cleaned" }, outputs: { output_type: "dataframe", resulting_state: "typed" } },
  { cell_id: "RENAME_COLS", stage: 1, type: "micro", keywords: ["rename","column","header","schema"], inputs: { input_type: "dataframe", expected_state: "raw_loaded" }, outputs: { output_type: "dataframe", resulting_state: "normalized" } },
  // Stage 2 — Transform
  { cell_id: "FILTER_ROWS", stage: 2, type: "micro", keywords: ["filter","where","condition","select","query"], inputs: { input_type: "dataframe", expected_state: "typed" }, outputs: { output_type: "dataframe", resulting_state: "filtered" } },
  { cell_id: "SORT_DATA", stage: 2, type: "micro", keywords: ["sort","order","rank","ascending","descending"], inputs: { input_type: "dataframe", expected_state: "typed" }, outputs: { output_type: "dataframe", resulting_state: "sorted" } },
  { cell_id: "NORMALIZE", stage: 2, type: "micro", keywords: ["normalize","scale","minmax","standardize","zscore"], inputs: { input_type: "dataframe", expected_state: "typed" }, outputs: { output_type: "dataframe", resulting_state: "scaled" } },
  { cell_id: "ENCODE_CATEGORICAL", stage: 2, type: "micro", keywords: ["encode","categorical","onehot","label","ordinal"], inputs: { input_type: "dataframe", expected_state: "typed" }, outputs: { output_type: "dataframe", resulting_state: "encoded" } },
  { cell_id: "MERGE_JOIN", stage: 2, type: "micro", keywords: ["merge","join","combine","union","concat"], inputs: { input_type: "dataframe", expected_state: "typed" }, outputs: { output_type: "dataframe", resulting_state: "merged" } },
  // Stage 3 — Aggregate
  { cell_id: "GROUP_BY", stage: 3, type: "micro", keywords: ["group","aggregate","groupby","sum","count","mean"], inputs: { input_type: "dataframe", expected_state: "filtered" }, outputs: { output_type: "dataframe", resulting_state: "aggregated" } },
  { cell_id: "PIVOT_TABLE", stage: 3, type: "micro", keywords: ["pivot","crosstab","table","reshape","wide"], inputs: { input_type: "dataframe", expected_state: "filtered" }, outputs: { output_type: "dataframe", resulting_state: "pivoted" } },
  { cell_id: "WINDOW_FUNC", stage: 3, type: "micro", keywords: ["window","rolling","cumsum","lag","lead"], inputs: { input_type: "dataframe", expected_state: "sorted" }, outputs: { output_type: "dataframe", resulting_state: "windowed" } },
  { cell_id: "FEATURE_ENG", stage: 3, type: "micro", keywords: ["feature","engineer","derive","compute","create"], inputs: { input_type: "dataframe", expected_state: "scaled" }, outputs: { output_type: "dataframe", resulting_state: "featured" } },
  // Stage 4 — Analyze
  { cell_id: "DESCRIBE_STATS", stage: 4, type: "micro", keywords: ["stats","describe","summary","distribution","mean","std"], inputs: { input_type: "dataframe", expected_state: "aggregated" }, outputs: { output_type: "report", resulting_state: "stats_report" } },
  { cell_id: "CORRELATION", stage: 4, type: "micro", keywords: ["correlation","pearson","spearman","heatmap","matrix"], inputs: { input_type: "dataframe", expected_state: "featured" }, outputs: { output_type: "report", resulting_state: "corr_report" } },
  { cell_id: "OUTLIER_DETECT", stage: 4, type: "micro", keywords: ["outlier","anomaly","detect","iqr","zscore"], inputs: { input_type: "dataframe", expected_state: "aggregated" }, outputs: { output_type: "report", resulting_state: "outlier_report" } },
  { cell_id: "TRAIN_MODEL", stage: 4, type: "micro", keywords: ["train","model","fit","ml","sklearn","classifier","regressor"], inputs: { input_type: "dataframe", expected_state: "featured" }, outputs: { output_type: "model", resulting_state: "trained_model" } },
  // Stage 5 — Export
  { cell_id: "EXPORT_CSV", stage: 5, type: "micro", keywords: ["export","save","csv","write","output"], inputs: { input_type: "dataframe", expected_state: "aggregated" }, outputs: { output_type: "file", resulting_state: "csv_exported" } },
  { cell_id: "EXPORT_JSON", stage: 5, type: "micro", keywords: ["export","json","save","write","output"], inputs: { input_type: "dataframe", expected_state: "aggregated" }, outputs: { output_type: "file", resulting_state: "json_exported" } },
  { cell_id: "EXPORT_HTML", stage: 5, type: "micro", keywords: ["export","html","report","render","visualize"], inputs: { input_type: "report", expected_state: "stats_report" }, outputs: { output_type: "file", resulting_state: "html_exported" } },
  { cell_id: "EXPORT_PARQUET", stage: 5, type: "micro", keywords: ["export","parquet","feather","fast","binary"], inputs: { input_type: "dataframe", expected_state: "aggregated" }, outputs: { output_type: "file", resulting_state: "parquet_exported" } },
  { cell_id: "SERVE_API", stage: 5, type: "micro", keywords: ["serve","api","flask","fastapi","endpoint","rest"], inputs: { input_type: "model", expected_state: "trained_model" }, outputs: { output_type: "service", resulting_state: "api_served" } },
  // Macros
  { cell_id: "FULL_ETL_PIPELINE", stage: 0, type: "macro", keywords: ["etl","pipeline","full","end-to-end","complete"], inputs: {}, outputs: {}, intent_expansion: ["load csv", "clean nulls", "filter rows", "group by", "export csv"] },
  { cell_id: "ML_PIPELINE", stage: 0, type: "macro", keywords: ["machine learning","ml","train","predict","model"], inputs: {}, outputs: {}, intent_expansion: ["load csv", "type cast", "normalize data", "engineer features", "train model", "serve api"] },
];

const CODE_TEMPLATES = {
  LOAD_CSV: `{output_var} = pd.read_csv('{input_var}')`,
  LOAD_JSON: `{output_var} = pd.read_json('{input_var}')`,
  LOAD_EXCEL: `{output_var} = pd.read_excel('{input_var}')`,
  LOAD_PARQUET: `{output_var} = pd.read_parquet('{input_var}')`,
  STREAM_API: `import requests\nresp = requests.get('{input_var}')\n{output_var} = pd.DataFrame(resp.json())`,
  DROP_NULLS: `{output_var} = {input_var}.dropna()`,
  FILL_NULLS: `{output_var} = {input_var}.fillna(method='ffill').fillna(method='bfill')`,
  DEDUPLICATE: `{output_var} = {input_var}.drop_duplicates()`,
  TYPE_CAST: `{output_var} = {input_var}.infer_objects()`,
  RENAME_COLS: `{output_var} = {input_var}.rename(columns=str.lower)`,
  FILTER_ROWS: `{output_var} = {input_var}[{input_var}.notnull().all(axis=1)]`,
  SORT_DATA: `{output_var} = {input_var}.sort_values(by={input_var}.columns[0])`,
  NORMALIZE: `from sklearn.preprocessing import MinMaxScaler\n_scaler = MinMaxScaler()\n{output_var} = pd.DataFrame(_scaler.fit_transform({input_var}), columns={input_var}.columns)`,
  ENCODE_CATEGORICAL: `{output_var} = pd.get_dummies({input_var})`,
  MERGE_JOIN: `{output_var} = {input_var}.copy()  # merge with secondary source`,
  GROUP_BY: `{output_var} = {input_var}.groupby({input_var}.columns[0]).sum().reset_index()`,
  PIVOT_TABLE: `{output_var} = {input_var}.pivot_table(index={input_var}.columns[0])`,
  WINDOW_FUNC: `{output_var} = {input_var}.assign(rolling_mean={input_var}.iloc[:,0].rolling(3).mean())`,
  FEATURE_ENG: `{output_var} = {input_var}.copy()\n{output_var}['_feature_ratio'] = {input_var}.iloc[:,0] / ({input_var}.iloc[:,1] + 1e-9)`,
  DESCRIBE_STATS: `{output_var} = {input_var}.describe().to_dict()`,
  CORRELATION: `{output_var} = {input_var}.corr().to_dict()`,
  OUTLIER_DETECT: `_q1 = {input_var}.quantile(0.25)\n_q3 = {input_var}.quantile(0.75)\n_iqr = _q3 - _q1\n{output_var} = ((({input_var} < (_q1 - 1.5*_iqr)) | ({input_var} > (_q3 + 1.5*_iqr)))).sum().to_dict()`,
  TRAIN_MODEL: `from sklearn.ensemble import RandomForestClassifier\n_clf = RandomForestClassifier()\n_X = {input_var}.iloc[:,:-1]\n_y = {input_var}.iloc[:,-1]\n_clf.fit(_X, _y)\n{output_var} = _clf`,
  EXPORT_CSV: `{input_var}.to_csv('{output_var}', index=False)`,
  EXPORT_JSON: `{input_var}.to_json('{output_var}', orient='records', indent=2)`,
  EXPORT_HTML: `import json\nwith open('{output_var}', 'w') as f:\n    f.write('<pre>' + json.dumps({input_var}, indent=2) + '</pre>')`,
  EXPORT_PARQUET: `{input_var}.to_parquet('{output_var}', index=False)`,
  SERVE_API: `import pickle\nwith open('{output_var}', 'wb') as f:\n    pickle.dump({input_var}, f)`,
};

// ─── ROUTING ENGINE (JS port of LatticeRouter) ───────────────────────────────
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

function keywordEmbed(text) {
  const stopwords = new Set(["a","an","the","and","or","but","for","with","to","of","in","on"]);
  const words = text.toLowerCase().replace(/[^a-z0-9 ]/g,"").split(" ").filter(w => w && !stopwords.has(w));
  const vocab = ["load","read","csv","json","excel","parquet","stream","api","clean","null","missing",
    "drop","fill","impute","dedup","type","cast","rename","filter","sort","normalize","encode",
    "merge","join","group","aggregate","pivot","window","feature","engineer","stats","describe",
    "correlation","outlier","detect","train","model","export","save","serve","endpoint","ml",
    "machine","learning","pipeline","etl","complete","end","full","binary","file","write","output"];
  return vocab.map(v => words.some(w => v.includes(w) || w.includes(v)) ? 1 : 0);
}

function cellEmbedding(cell) {
  const profile = `${cell.cell_id} ${cell.keywords.join(" ")}`.toLowerCase();
  return keywordEmbed(profile);
}

function scoreCell(cell, goalEmbed) {
  return cosineSim(goalEmbed, cellEmbedding(cell));
}

function planPath(goals, cells, logFn) {
  const micro = cells.filter(c => c.type === "micro");
  const macros = cells.filter(c => c.type === "macro");

  let rawGoals = goals.split(/,|\band\b|\bthen\b/i).map(g => g.trim()).filter(Boolean);
  const path = [];
  const virtualEdges = new Set();
  const expandedMacros = new Set();
  let currentType = "str";
  let currentState = "source_identifier";
  let currentNode = null;

  const MAX_GOALS = 60;
  const MIN_CONF = 0.25;
  const TUNNEL_MARGIN = 0.15;
  const MACRO_THRESH = 0.40;

  let step = 0;
  let iter = 0;
  while (rawGoals.length > 0 && iter < 200) {
    iter++;
    if (rawGoals.length > MAX_GOALS) { rawGoals = rawGoals.slice(0, MAX_GOALS); }
    const goal = rawGoals.shift();
    const goalKey = goal.toLowerCase().trim();
    const goalEmbed = keywordEmbed(goal);

    // Check macro
    let bestMacro = null, bestMacroScore = -1;
    for (const m of macros) {
      const s = scoreCell(m, goalEmbed);
      if (s > bestMacroScore) { bestMacroScore = s; bestMacro = m; }
    }
    let bestGlobalMicro = null, bestGlobalMicroScore = -1;
    for (const c of micro) {
      const s = scoreCell(c, goalEmbed);
      if (s > bestGlobalMicroScore) { bestGlobalMicroScore = s; bestGlobalMicro = c; }
    }

    if (bestMacro && bestMacroScore > MACRO_THRESH && bestGlobalMicroScore < 0.70 && !expandedMacros.has(goalKey)) {
      expandedMacros.add(goalKey);
      logFn(`[FRACTAL UNFOLDING] '${goal}' → ${bestMacro.intent_expansion.length} sub-ops`, "expand");
      rawGoals = [...bestMacro.intent_expansion, ...rawGoals];
      continue;
    }

    if (step === 0) {
      const cands = micro.filter(c => c.inputs.input_type === currentType && c.inputs.expected_state === currentState);
      let best = null, bestScore = -1;
      for (const c of cands) { const s = scoreCell(c, goalEmbed); if (s > bestScore) { bestScore = s; best = c; } }
      if (!best || bestScore < MIN_CONF) {
        // relax constraint
        for (const c of micro) { const s = scoreCell(c, goalEmbed); if (s > bestScore) { bestScore = s; best = c; } }
      }
      if (best) {
        logFn(`[ENTRY] '${goal}' → ${best.cell_id} (${bestScore.toFixed(2)})`, "route");
        path.push(best);
        currentNode = best;
        currentType = best.outputs.output_type;
        currentState = best.outputs.resulting_state;
        step++;
      } else {
        logFn(`[HALT] Entry confidence too low for '${goal}'`, "warn");
      }
      continue;
    }

    // Local neighbors
    const localCands = currentNode ? micro.filter(c =>
      c.inputs.input_type === currentNode.outputs.output_type &&
      c.inputs.expected_state === currentNode.outputs.resulting_state
    ) : [];
    let bestLocal = null, bestLocalScore = -1;
    for (const c of localCands) { const s = scoreCell(c, goalEmbed); if (s > bestLocalScore) { bestLocalScore = s; bestLocal = c; } }

    if (bestGlobalMicroScore > MIN_CONF && (bestLocalScore < MIN_CONF || bestGlobalMicroScore - bestLocalScore > TUNNEL_MARGIN)) {
      const targetType = bestGlobalMicro.inputs.input_type;
      if (targetType === currentType) {
        logFn(`[TUNNEL] '${goal}' → ${bestGlobalMicro.cell_id} (${bestGlobalMicroScore.toFixed(2)})`, "tunnel");
        virtualEdges.add(bestGlobalMicro.cell_id);
        path.push(bestGlobalMicro);
        currentNode = bestGlobalMicro;
        currentType = bestGlobalMicro.outputs.output_type;
        currentState = bestGlobalMicro.outputs.resulting_state;
        step++;
      } else {
        logFn(`[TYPE GAP] ${currentType} → ${targetType}, using local fallback`, "warn");
        if (bestLocal && bestLocalScore >= MIN_CONF) {
          path.push(bestLocal);
          currentNode = bestLocal;
          currentType = bestLocal.outputs.output_type;
          currentState = bestLocal.outputs.resulting_state;
          step++;
        } else { logFn(`[SKIP] No path for '${goal}'`, "warn"); step++; }
      }
    } else if (bestLocal && bestLocalScore >= MIN_CONF) {
      logFn(`[ROUTE] '${goal}' → ${bestLocal.cell_id} (${bestLocalScore.toFixed(2)})`, "route");
      path.push(bestLocal);
      currentNode = bestLocal;
      currentType = bestLocal.outputs.output_type;
      currentState = bestLocal.outputs.resulting_state;
      step++;
    } else {
      logFn(`[SKIP] No viable path for '${goal}'`, "warn");
      step++;
    }
  }

  logFn(`[COMPLETE] Path: [${path.map(c => c.cell_id).join(" → ")}]`, "success");
  return { path, virtualEdges };
}

function compileCode(path, prompt) {
  const registry = { input_source: { type: "str", state: "source_identifier" } };
  const varCounts = {};
  const lines = ["import pandas as pd", "import numpy as np", ""];

  const nameExtract = prompt.match(/["\']([^"\']+)["\']/) ||
    prompt.match(/\b([\w\-_.]+\.(?:csv|json|xlsx|parquet|feather|html))\b/i);
  const filename = nameExtract ? nameExtract[1] : "data.csv";

  lines.push(`# Source: ${filename}`);
  lines.push(`input_source = "${filename}"`);
  lines.push("");

  let prevVar = "input_source";
  for (const cell of path) {
    const baseName = cell.outputs.resulting_state.toLowerCase().replace(/[^a-z0-9]/g, "_");
    varCounts[baseName] = (varCounts[baseName] || 0) + 1;
    const outVar = varCounts[baseName] > 1 ? `${baseName}_v${varCounts[baseName]}` : baseName;

    const tmpl = CODE_TEMPLATES[cell.cell_id] || `# ${cell.cell_id}\n{output_var} = {input_var}`;
    const code = tmpl.replace(/\{input_var\}/g, prevVar).replace(/\{output_var\}/g, outVar);
    lines.push(`# ── ${cell.cell_id} (Stage ${cell.stage}) ──`);
    lines.push(code);
    lines.push("");
    prevVar = outVar;
    registry[outVar] = { type: cell.outputs.output_type, state: cell.outputs.resulting_state };
  }

  return lines.join("\n");
}

// ─── 3D LATTICE COMPONENT ────────────────────────────────────────────────────
const STAGE_COLORS_HEX = ["#00e5ff", "#00bfae", "#7c4dff", "#ff6d00", "#00e676", "#ff4081", "#40c4ff", "#ffd740"];
const STAGE_NAMES = ["Ingest", "Clean", "Transform", "Aggregate", "Analyze", "Export", "Encode", "Output"];

function ThreeLattice({ cells, activePath, virtualEdges, onNodeHover, onNodeClick }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0, isDragging: false, prevX: 0, prevY: 0 });
  const orbitRef = useRef({ theta: 0.3, phi: 1.1, radius: 80, target: new THREE.Vector3(30, 0, 0) });
  const nodeMapRef = useRef({});
  const raycasterRef = useRef(new THREE.Raycaster());
  const meshesRef = useRef([]);
  const isBuiltRef = useRef(false);

  const activeIds = useMemo(() => new Set(activePath.map(c => c.cell_id)), [activePath]);

  // Cylindrical layout
  const nodePositions = useMemo(() => {
    const stageGroups = {};
    for (const cell of cells.filter(c => c.type === "micro")) {
      if (!stageGroups[cell.stage]) stageGroups[cell.stage] = [];
      stageGroups[cell.stage].push(cell);
    }
    const positions = {};
    const STAGE_X_GAP = 14;
    const RING_CAP = 7;
    const R_BASE = 5;
    const R_STEP = 5;

    for (const [stage, stageCells] of Object.entries(stageGroups)) {
      const xPos = parseInt(stage) * STAGE_X_GAP;
      stageCells.forEach((cell, idx) => {
        if (stageCells.length === 1) {
          positions[cell.cell_id] = new THREE.Vector3(xPos, 0, 0);
        } else {
          const ring = Math.floor(idx / RING_CAP);
          const slot = idx % RING_CAP;
          const ringN = Math.min(RING_CAP, stageCells.length - ring * RING_CAP);
          const angle = (slot / ringN) * 2 * Math.PI + ring * (Math.PI / RING_CAP);
          const radius = R_BASE + ring * R_STEP;
          positions[cell.cell_id] = new THREE.Vector3(xPos, radius * Math.cos(angle), radius * Math.sin(angle));
        }
      });
    }
    return positions;
  }, [cells]);

  useEffect(() => {
    if (!mountRef.current) return;
    const W = mountRef.current.clientWidth;
    const H = mountRef.current.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x040810);
    scene.fog = new THREE.FogExp2(0x040810, 0.006);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 500);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.shadowMap.enabled = false;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Ambient + directional light
    scene.add(new THREE.AmbientLight(0x112244, 2.5));
    const dir = new THREE.DirectionalLight(0x4488ff, 1.5);
    dir.position.set(50, 30, 20);
    scene.add(dir);

    // Grid helper (subtle)
    const grid = new THREE.GridHelper(200, 30, 0x0a1a30, 0x0a1a30);
    grid.rotation.z = Math.PI / 2;
    grid.position.x = 42;
    scene.add(grid);

    isBuiltRef.current = false;

    // Animate loop
    let animFrame;
    const animate = () => {
      animFrame = requestAnimationFrame(animate);
      const { theta, phi, radius, target } = orbitRef.current;
      camera.position.x = target.x + radius * Math.sin(phi) * Math.cos(theta);
      camera.position.y = target.y + radius * Math.cos(phi);
      camera.position.z = target.z + radius * Math.sin(phi) * Math.sin(theta);
      camera.lookAt(target);
      renderer.render(scene, camera);
    };
    animate();
    frameRef.current = animFrame;

    // Resize
    const onResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth, h = mountRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animFrame);
      window.removeEventListener("resize", onResize);
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // Build/update scene geometry when cells or active path changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Remove old meshes
    for (const m of meshesRef.current) scene.remove(m);
    meshesRef.current = [];
    nodeMapRef.current = {};

    const microCells = cells.filter(c => c.type === "micro");
    const stages = [...new Set(microCells.map(c => c.stage))].sort((a, b) => a - b);

    // Stage zone planes
    stages.forEach((stage, i) => {
      const xPos = stage * 14;
      const planeGeo = new THREE.PlaneGeometry(2, 28);
      const planeMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(STAGE_COLORS_HEX[i % STAGE_COLORS_HEX.length]),
        transparent: true, opacity: 0.04, side: THREE.DoubleSide, depthWrite: false
      });
      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.rotation.y = Math.PI / 2;
      plane.position.set(xPos, 0, 0);
      scene.add(plane);
      meshesRef.current.push(plane);
    });

    // Dormant edges (limited to adjacent stages)
    const edgeGeo = new THREE.BufferGeometry();
    const edgeVerts = [];
    let edgeCount = 0;
    const MAX_EDGES = 300;

    const activeStages = new Set(activePath.map(c => c.stage));
    const nearStages = new Set();
    for (const s of activeStages) { nearStages.add(s-1); nearStages.add(s); nearStages.add(s+1); }
    const drawStages = nearStages.size > 0 ? nearStages : new Set(stages);

    for (const ca of microCells) {
      if (edgeCount >= MAX_EDGES) break;
      if (!drawStages.has(ca.stage) || !nodePositions[ca.cell_id]) continue;
      for (const cb of microCells) {
        if (!nodePositions[cb.cell_id]) continue;
        if (ca.outputs.output_type === cb.inputs.input_type && ca.outputs.resulting_state === cb.inputs.expected_state) {
          const pa = nodePositions[ca.cell_id], pb = nodePositions[cb.cell_id];
          edgeVerts.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
          edgeCount++;
          if (edgeCount >= MAX_EDGES) break;
        }
      }
    }

    if (edgeVerts.length > 0) {
      edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(edgeVerts, 3));
      const edgeMat = new THREE.LineBasicMaterial({ color: 0x0d1f4a, transparent: true, opacity: 0.35 });
      const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
      scene.add(edgeLines);
      meshesRef.current.push(edgeLines);
    }

    // Active path edges
    if (activePath.length > 1) {
      const actVerts = [], tunVerts = [];
      for (let i = 0; i < activePath.length - 1; i++) {
        const pa = nodePositions[activePath[i].cell_id];
        const pb = nodePositions[activePath[i+1].cell_id];
        if (!pa || !pb) continue;
        if (virtualEdges.has(activePath[i+1].cell_id)) {
          tunVerts.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
        } else {
          actVerts.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
        }
      }
      if (actVerts.length > 0) {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(actVerts, 3));
        const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 }));
        scene.add(l); meshesRef.current.push(l);
      }
      if (tunVerts.length > 0) {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(tunVerts, 3));
        const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xff00aa, linewidth: 2 }));
        scene.add(l); meshesRef.current.push(l);
      }
    }

    // Nodes
    const dormantGeo = new THREE.SphereGeometry(0.45, 8, 8);
    const activeGeo = new THREE.SphereGeometry(0.75, 12, 12);
    const glowGeo = new THREE.SphereGeometry(1.4, 8, 8);
    const tunnelGeo = new THREE.OctahedronGeometry(0.85);

    for (const cell of microCells) {
      const pos = nodePositions[cell.cell_id];
      if (!pos) continue;

      const isActive = activeIds.has(cell.cell_id);
      const isTunnel = virtualEdges.has(cell.cell_id);

      if (isTunnel) {
        // Glow halo
        const halo = new THREE.Mesh(glowGeo, new THREE.MeshBasicMaterial({ color: 0xff00aa, transparent: true, opacity: 0.12 }));
        halo.position.copy(pos);
        scene.add(halo); meshesRef.current.push(halo);
        // Core octahedron
        const core = new THREE.Mesh(tunnelGeo, new THREE.MeshStandardMaterial({ color: 0xff00aa, emissive: 0xff00aa, emissiveIntensity: 0.6, roughness: 0.2, metalness: 0.5 }));
        core.position.copy(pos);
        core.userData = { cell, isTunnel: true };
        scene.add(core); meshesRef.current.push(core);
        nodeMapRef.current[cell.cell_id] = core;
      } else if (isActive) {
        // Double glow
        const halo1 = new THREE.Mesh(new THREE.SphereGeometry(2.0, 8, 8), new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.05 }));
        halo1.position.copy(pos); scene.add(halo1); meshesRef.current.push(halo1);
        const halo2 = new THREE.Mesh(glowGeo, new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.12 }));
        halo2.position.copy(pos); scene.add(halo2); meshesRef.current.push(halo2);
        // Core sphere
        const core = new THREE.Mesh(activeGeo, new THREE.MeshStandardMaterial({ color: 0x00e5ff, emissive: 0x00e5ff, emissiveIntensity: 0.7, roughness: 0.15, metalness: 0.6 }));
        core.position.copy(pos);
        core.userData = { cell, isActive: true };
        scene.add(core); meshesRef.current.push(core);
        nodeMapRef.current[cell.cell_id] = core;
      } else {
        const stageIdx = stages.indexOf(cell.stage);
        const col = new THREE.Color(STAGE_COLORS_HEX[stageIdx % STAGE_COLORS_HEX.length]).multiplyScalar(0.35);
        const node = new THREE.Mesh(dormantGeo, new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.2, roughness: 0.6, metalness: 0.2, transparent: true, opacity: 0.55 }));
        node.position.copy(pos);
        node.userData = { cell };
        scene.add(node); meshesRef.current.push(node);
        nodeMapRef.current[cell.cell_id] = node;
      }
    }

    // Flow cones on active path
    for (let i = 0; i < activePath.length - 1; i++) {
      const pa = nodePositions[activePath[i].cell_id];
      const pb = nodePositions[activePath[i+1].cell_id];
      if (!pa || !pb) continue;
      const dir = pb.clone().sub(pa).normalize();
      const mid = pa.clone().lerp(pb, 0.62);
      const coneGeo = new THREE.ConeGeometry(0.35, 1.0, 8);
      const coneMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.85 });
      const cone = new THREE.Mesh(coneGeo, coneMat);
      cone.position.copy(mid);
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      scene.add(cone); meshesRef.current.push(cone);
    }

  }, [cells, activePath, virtualEdges, nodePositions, activeIds]);

  // Mouse orbit + hover
  const handleMouseDown = useCallback(e => {
    mouseRef.current.isDragging = true;
    mouseRef.current.prevX = e.clientX;
    mouseRef.current.prevY = e.clientY;
  }, []);

  const handleMouseUp = useCallback(() => { mouseRef.current.isDragging = false; }, []);

  const handleMouseMove = useCallback(e => {
    const r = mountRef.current?.getBoundingClientRect();
    if (!r) return;
    const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
    const ny = -((e.clientY - r.top) / r.height) * 2 + 1;
    mouseRef.current.x = nx;
    mouseRef.current.y = ny;

    if (mouseRef.current.isDragging) {
      const dx = e.clientX - mouseRef.current.prevX;
      const dy = e.clientY - mouseRef.current.prevY;
      orbitRef.current.theta -= dx * 0.008;
      orbitRef.current.phi = Math.max(0.2, Math.min(Math.PI - 0.2, orbitRef.current.phi + dy * 0.008));
      mouseRef.current.prevX = e.clientX;
      mouseRef.current.prevY = e.clientY;
    } else {
      // Hover
      const cam = cameraRef.current, scene = sceneRef.current;
      if (!cam || !scene) return;
      raycasterRef.current.setFromCamera({ x: nx, y: ny }, cam);
      const clickable = Object.values(nodeMapRef.current);
      const hits = raycasterRef.current.intersectObjects(clickable);
      if (hits.length > 0 && hits[0].object.userData?.cell) {
        onNodeHover(hits[0].object.userData.cell, { x: e.clientX, y: e.clientY });
        mountRef.current.style.cursor = "pointer";
      } else {
        onNodeHover(null, null);
        mountRef.current.style.cursor = "grab";
      }
    }
  }, [onNodeHover]);

  const handleWheel = useCallback(e => {
    orbitRef.current.radius = Math.max(15, Math.min(200, orbitRef.current.radius + e.deltaY * 0.08));
  }, []);

  const handleClick = useCallback(e => {
    const cam = cameraRef.current, scene = sceneRef.current;
    if (!cam || !scene) return;
    const r = mountRef.current?.getBoundingClientRect();
    if (!r) return;
    const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
    const ny = -((e.clientY - r.top) / r.height) * 2 + 1;
    raycasterRef.current.setFromCamera({ x: nx, y: ny }, cam);
    const clickable = Object.values(nodeMapRef.current);
    const hits = raycasterRef.current.intersectObjects(clickable);
    if (hits.length > 0 && hits[0].object.userData?.cell) {
      onNodeClick(hits[0].object.userData.cell);
    }
  }, [onNodeClick]);

  return (
    <div
      ref={mountRef}
      className="w-full h-full"
      style={{ cursor: "grab" }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onMouseMove={handleMouseMove}
      onWheel={handleWheel}
      onClick={handleClick}
    />
  );
}

// ─── SYNTAX HIGHLIGHTER (lightweight, Python-aware) ──────────────────────────
function SyntaxLine({ line }) {
  const tokens = [];
  const KEYWORDS = /\b(import|from|as|def|class|return|if|else|elif|for|while|with|in|not|and|or|True|False|None)\b/g;
  const STRINGS = /(["'])(?:\\.|[^\\])*?\1/g;
  const COMMENTS = /(#.*)/g;
  const NUMBERS = /\b(\d+(?:\.\d+)?)\b/g;
  const BUILTINS = /\b(pd|np|print|len|range|list|dict|str|int|float|open|type)\b/g;

  let result = line
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  result = result
    .replace(COMMENTS, '<span style="color:#5c6e8a">$1</span>')
    .replace(STRINGS, '<span style="color:#98c379">$&</span>')
    .replace(KEYWORDS, '<span style="color:#c678dd">$1</span>')
    .replace(BUILTINS, '<span style="color:#61afef">$1</span>')
    .replace(NUMBERS, '<span style="color:#d19a66">$1</span>');

  return <div className="leading-relaxed" dangerouslySetInnerHTML={{ __html: result || "&nbsp;" }} />;
}

// ─── LOG ENTRY COMPONENT ───────────────────────────────────────────────────────
const LOG_STYLES = {
  route: { color: "#61afef", icon: "→" },
  tunnel: { color: "#c678dd", icon: "⤳" },
  expand: { color: "#e5c07b", icon: "⊞" },
  warn: { color: "#e06c75", icon: "⚠" },
  success: { color: "#98c379", icon: "✓" },
  info: { color: "#5c6e8a", icon: "·" },
  system: { color: "#4a5568", icon: "◆" },
};

function LogEntry({ entry }) {
  const s = LOG_STYLES[entry.type] || LOG_STYLES.info;
  return (
    <div className="flex gap-2 py-0.5 font-mono text-xs leading-relaxed">
      <span style={{ color: s.color, minWidth: 14 }}>{s.icon}</span>
      <span style={{ color: "#4a5568", minWidth: 60 }}>{entry.time}</span>
      <span style={{ color: s.color }}>{entry.msg}</span>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function NSTLApp() {
  const [prompt, setPrompt] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [activePath, setActivePath] = useState([]);
  const [virtualEdges, setVirtualEdges] = useState(new Set());
  const [generatedCode, setGeneratedCode] = useState("# Generated code will appear here after running a prompt.\n");
  const [logs, setLogs] = useState([
    { type: "system", msg: "NSTL Engine initialized. 28 micro-cells loaded, 2 macros.", time: "00:00:00" },
    { type: "system", msg: "Semantic manifolds pre-computed. Router ready.", time: "00:00:00" },
  ]);
  const [cellFilter, setCellFilter] = useState("");
  const [expandedStages, setExpandedStages] = useState(new Set([0,1,2]));
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoverPos, setHoverPos] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [rightTab, setRightTab] = useState("code"); // "code" | "inspect"
  const [activeTab, setActiveTab] = useState("lattice"); // "lattice" | "settings"
  const [bottomTab, setBottomTab] = useState("log"); // "log" | "path"
  const [copied, setCopied] = useState(false);
  const logEndRef = useRef(null);
  const textareaRef = useRef(null);

  const addLog = useCallback((msg, type = "info") => {
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}:${String(now.getSeconds()).padStart(2,"0")}`;
    setLogs(prev => [...prev.slice(-200), { msg, type, time }]);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const microCells = MOCK_CELLS.filter(c => c.type === "micro");
  const stageGroups = useMemo(() => {
    const g = {};
    for (const c of microCells) {
      if (!g[c.stage]) g[c.stage] = [];
      g[c.stage].push(c);
    }
    return g;
  }, []);

  const filteredGroups = useMemo(() => {
    if (!cellFilter) return stageGroups;
    const q = cellFilter.toLowerCase();
    const result = {};
    for (const [stage, cs] of Object.entries(stageGroups)) {
      const filtered = cs.filter(c => c.cell_id.toLowerCase().includes(q) || c.keywords.some(k => k.includes(q)));
      if (filtered.length > 0) result[stage] = filtered;
    }
    return result;
  }, [cellFilter, stageGroups]);

  const handleRun = useCallback(() => {
    if (!prompt.trim() || isRunning) return;
    setIsRunning(true);
    setActivePath([]);
    setVirtualEdges(new Set());
    addLog(`Prompt received: "${prompt}"`, "system");
    addLog("Neural routing initiated...", "info");

    setTimeout(() => {
      const { path, virtualEdges: ve } = planPath(prompt, MOCK_CELLS, addLog);
      setActivePath(path);
      setVirtualEdges(ve);
      const code = compileCode(path, prompt);
      setGeneratedCode(code);
      addLog(`Compiled ${path.length} cells into ${code.split("\n").length} lines`, "success");
      setIsRunning(false);
      setRightTab("code");
      setBottomTab("path");
    }, 600 + Math.random() * 400);
  }, [prompt, isRunning, addLog]);

  const handleKeyDown = useCallback(e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleRun(); }
  }, [handleRun]);

  const handleCopyCode = useCallback(() => {
    navigator.clipboard?.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, [generatedCode]);

  const handleNodeHover = useCallback((cell, pos) => {
    setHoveredNode(cell);
    setHoverPos(pos);
  }, []);

  const handleNodeClick = useCallback(cell => {
    setSelectedNode(cell);
    setRightTab("inspect");
  }, []);

  const toggleStage = useCallback(stage => {
    setExpandedStages(prev => {
      const n = new Set(prev);
      if (n.has(stage)) n.delete(stage); else n.add(stage);
      return n;
    });
  }, []);

  const codeLines = generatedCode.split("\n");

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: "#060b14", fontFamily: "'JetBrains Mono', 'Fira Code', monospace", color: "#ccd6f6" }}>

      {/* ── TOP BAR ── */}
      <div className="flex items-center justify-between px-4 py-2 border-b" style={{ borderColor: "#0d1f3c", background: "#030712", minHeight: 44 }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Network size={16} style={{ color: "#00e5ff" }} />
            <span className="text-sm font-bold tracking-widest" style={{ color: "#00e5ff", fontSize: 11 }}>NSTL</span>
            <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>CYBER-LATTICE ENGINE</span>
          </div>
          <div className="h-4 w-px" style={{ background: "#0d1f3c" }} />
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: "#98c379" }} />
            <span className="text-xs" style={{ color: "#4a5568", fontSize: 10 }}>ONLINE</span>
          </div>
          <div className="flex items-center gap-1">
            <Cpu size={11} style={{ color: "#4a5568" }} />
            <span className="text-xs" style={{ color: "#4a5568", fontSize: 10 }}>{microCells.length} CELLS</span>
          </div>
          <div className="flex items-center gap-1">
            <GitBranch size={11} style={{ color: "#4a5568" }} />
            <span className="text-xs" style={{ color: "#4a5568", fontSize: 10 }}>all-MiniLM-L6-v2</span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {/* Tab buttons */}
          {[{ id: "lattice", icon: <Layers size={12}/>, label: "Lattice" }, { id: "settings", icon: <Settings size={12}/>, label: "Settings" }].map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className="flex items-center gap-1 px-3 py-1 rounded text-xs transition-all"
              style={{ background: activeTab === t.id ? "#0d1f3c" : "transparent", color: activeTab === t.id ? "#00e5ff" : "#4a5568", border: "1px solid", borderColor: activeTab === t.id ? "#00e5ff33" : "transparent", fontSize: 10 }}>
              {t.icon}{t.label}
            </button>
          ))}
          {/* Upload stub */}
          <button className="flex items-center gap-1 px-3 py-1 rounded text-xs transition-all"
            style={{ background: "transparent", color: "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}
            title="Upload trees/ folder (future feature)">
            <Upload size={11} /><span>Trees</span>
          </button>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono" style={{ color: "#2a4a6a", fontSize: 10 }}>Matthew Nader · 202300103</span>
        </div>
      </div>

      {/* ── MAIN CONTENT ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── LEFT SIDEBAR — Cell Browser ── */}
        <div className="flex flex-col border-r" style={{ borderColor: "#0d1f3c", background: "#040a14", width: 220, minWidth: 180 }}>
          <div className="px-3 py-2 border-b flex items-center gap-2" style={{ borderColor: "#0d1f3c" }}>
            <Database size={12} style={{ color: "#4a5568" }} />
            <span className="text-xs font-bold tracking-widest" style={{ color: "#2a4a6a", fontSize: 10 }}>CELL BROWSER</span>
          </div>
          {/* Search */}
          <div className="px-2 py-2 border-b" style={{ borderColor: "#0d1f3c" }}>
            <div className="flex items-center gap-2 rounded px-2" style={{ background: "#060e1c", border: "1px solid #0d1f3c" }}>
              <Search size={11} style={{ color: "#2a4a6a" }} />
              <input
                className="flex-1 bg-transparent text-xs outline-none py-1.5"
                style={{ color: "#8892b0", fontSize: 11 }}
                placeholder="filter cells..."
                value={cellFilter}
                onChange={e => setCellFilter(e.target.value)}
              />
            </div>
          </div>
          {/* Stage groups */}
          <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: "thin", scrollbarColor: "#0d1f3c transparent" }}>
            {Object.entries(filteredGroups).sort(([a],[b]) => a-b).map(([stage, cs]) => {
              const si = parseInt(stage);
              const isExpanded = expandedStages.has(si);
              const stageColor = STAGE_COLORS_HEX[si % STAGE_COLORS_HEX.length];
              return (
                <div key={stage}>
                  <button onClick={() => toggleStage(si)}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-left transition-all hover:opacity-80"
                    style={{ background: "#060e1c" }}>
                    {isExpanded ? <ChevronDown size={10} style={{ color: "#2a4a6a" }} /> : <ChevronRight size={10} style={{ color: "#2a4a6a" }} />}
                    <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: stageColor, opacity: 0.8 }} />
                    <span className="text-xs font-bold" style={{ color: "#4a5568", fontSize: 10 }}>S{si}: {STAGE_NAMES[si] || "Stage"}</span>
                    <span className="ml-auto text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>{cs.length}</span>
                  </button>
                  {isExpanded && cs.map(cell => {
                    const isActive = activeIds.has(cell.cell_id);
                    const isTunnel = virtualEdges.has(cell.cell_id);
                    return (
                      <button key={cell.cell_id}
                        onClick={() => { setSelectedNode(cell); setRightTab("inspect"); }}
                        className="w-full flex items-center gap-2 px-4 py-1.5 text-left hover:opacity-90 transition-all"
                        style={{ background: isActive ? "#0a1a2e" : isTunnel ? "#140a1c" : "transparent" }}>
                        <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{
                          background: isTunnel ? "#ff00aa" : isActive ? "#00e5ff" : stageColor,
                          opacity: isActive || isTunnel ? 1 : 0.4,
                          boxShadow: isActive ? "0 0 6px #00e5ff" : isTunnel ? "0 0 6px #ff00aa" : "none"
                        }} />
                        <span className="text-xs truncate" style={{ color: isActive ? "#00e5ff" : isTunnel ? "#ff00aa" : "#4a5568", fontSize: 10, fontWeight: isActive ? "bold" : "normal" }}>
                          {cell.cell_id}
                        </span>
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>
          {/* Path stats */}
          {activePath.length > 0 && (
            <div className="px-3 py-2 border-t" style={{ borderColor: "#0d1f3c" }}>
              <div className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>ACTIVE PATH</div>
              <div className="flex items-center gap-2 mt-1">
                <Zap size={11} style={{ color: "#00e5ff" }} />
                <span className="text-xs" style={{ color: "#00e5ff", fontSize: 11 }}>{activePath.length} cells</span>
                {virtualEdges.size > 0 && (
                  <span className="text-xs" style={{ color: "#ff00aa", fontSize: 10 }}>{virtualEdges.size} tunnels</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── CENTER — 3D View + Prompt ── */}
        <div className="flex flex-col flex-1 overflow-hidden">

          {/* 3D Lattice view */}
          <div className="relative flex-1 overflow-hidden" style={{ minHeight: 200 }}>
            {activeTab === "lattice" ? (
              <>
                <ThreeLattice
                  cells={MOCK_CELLS}
                  activePath={activePath}
                  virtualEdges={virtualEdges}
                  onNodeHover={handleNodeHover}
                  onNodeClick={handleNodeClick}
                />
                {/* Legend overlay */}
                <div className="absolute bottom-4 left-4 flex flex-col gap-1.5 pointer-events-none">
                  {[
                    { color: "#00e5ff", label: "Active node", glow: true },
                    { color: "#ff00aa", label: "Tunnel node", glow: true },
                    { color: "#192850", label: "Dormant node", glow: false },
                    { color: "#00e5ff", label: "Execution path", line: true },
                    { color: "#ff00aa", label: "Semantic tunnel", line: true, dashed: true },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-2">
                      {item.line ? (
                        <div style={{ width: 20, height: 2, background: item.color, opacity: 0.8, borderBottom: item.dashed ? `2px dashed ${item.color}` : undefined, background: item.dashed ? "none" : item.color }} />
                      ) : (
                        <div className="w-2 h-2 rounded-full" style={{ background: item.color, boxShadow: item.glow ? `0 0 6px ${item.color}` : "none" }} />
                      )}
                      <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 9 }}>{item.label}</span>
                    </div>
                  ))}
                </div>
                {/* Controls hint */}
                <div className="absolute top-3 right-3 text-right pointer-events-none">
                  <div className="text-xs" style={{ color: "#1a2a3a", fontSize: 9 }}>Drag to orbit · Scroll to zoom · Click to inspect</div>
                </div>
                {/* Stage labels overlay */}
                {activePath.length > 0 && (
                  <div className="absolute top-3 left-3 pointer-events-none">
                    <div className="flex gap-1 flex-wrap" style={{ maxWidth: 300 }}>
                      {activePath.map((cell, i) => (
                        <div key={i} className="flex items-center gap-1 px-2 py-0.5 rounded"
                          style={{ background: "#060b1499", border: "1px solid #00e5ff22", fontSize: 9 }}>
                          <span style={{ color: "#2a5a7a" }}>{i+1}</span>
                          <span style={{ color: "#00c8e8" }}>{cell.cell_id}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              /* Settings Panel */
              <div className="p-6 overflow-y-auto h-full" style={{ scrollbarWidth: "thin", scrollbarColor: "#0d1f3c transparent" }}>
                <div className="text-xs font-bold mb-4 tracking-widest" style={{ color: "#2a4a6a", fontSize: 10 }}>ENGINE SETTINGS</div>
                <div className="grid gap-4" style={{ maxWidth: 500 }}>
                  {[
                    { label: "Min Confidence Threshold", val: "0.25", desc: "Minimum router confidence to accept a cell" },
                    { label: "Tunneling Margin", val: "0.15", desc: "Score gap required to trigger a semantic tunnel" },
                    { label: "Macro Expansion Threshold", val: "0.40", desc: "Score required to trigger macro unfolding" },
                    { label: "Micro Override Score", val: "0.70", desc: "Micro confidence that overrides macro expansion" },
                    { label: "Max Goal Queue", val: "60", desc: "Hard ceiling on goal expansion queue" },
                    { label: "Max Base Edges", val: "300", desc: "Performance cap on rendered lattice edges" },
                  ].map(s => (
                    <div key={s.label} className="rounded p-3" style={{ background: "#060e1c", border: "1px solid #0d1f3c" }}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-bold" style={{ color: "#8892b0", fontSize: 11 }}>{s.label}</span>
                        <input defaultValue={s.val} className="w-20 text-right bg-transparent outline-none text-xs px-1 py-0.5 rounded" style={{ color: "#00e5ff", border: "1px solid #0d1f3c", fontSize: 11 }} readOnly />
                      </div>
                      <div className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>{s.desc}</div>
                    </div>
                  ))}
                  <div className="rounded p-3" style={{ background: "#060e1c", border: "1px solid #0d1f3c" }}>
                    <div className="text-xs font-bold mb-2" style={{ color: "#8892b0", fontSize: 11 }}>Sentence Transformer Model</div>
                    <select className="w-full text-xs px-2 py-1 rounded" style={{ background: "#030712", color: "#8892b0", border: "1px solid #0d1f3c", fontSize: 11 }} disabled>
                      <option>all-MiniLM-L6-v2</option>
                      <option>all-mpnet-base-v2</option>
                      <option>paraphrase-MiniLM-L3-v2</option>
                    </select>
                    <div className="text-xs mt-1" style={{ color: "#2a4a6a", fontSize: 10 }}>Model reload requires engine restart</div>
                  </div>
                  <div className="flex gap-2 mt-2">
                    <button className="flex-1 py-2 rounded text-xs transition-all hover:opacity-80" style={{ background: "#0d1f3c", color: "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}>
                      Load Session
                    </button>
                    <button className="flex-1 py-2 rounded text-xs transition-all hover:opacity-80" style={{ background: "#0d1f3c", color: "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}>
                      Save Session
                    </button>
                    <button className="flex-1 py-2 rounded text-xs transition-all hover:opacity-80" style={{ background: "#0d1f3c", color: "#e06c75", border: "1px solid #1a0a0a", fontSize: 10 }}>
                      Reset Engine
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Node hover tooltip */}
            {hoveredNode && hoverPos && (
              <div className="fixed z-50 pointer-events-none rounded px-3 py-2" style={{
                left: hoverPos.x + 14, top: hoverPos.y - 8, background: "#030712ee",
                border: "1px solid #0d1f3c", maxWidth: 240, fontSize: 11
              }}>
                <div className="font-bold mb-1" style={{ color: "#00e5ff", fontSize: 11 }}>{hoveredNode.cell_id}</div>
                <div style={{ color: "#4a5568", fontSize: 10 }}>Stage {hoveredNode.stage} · {STAGE_NAMES[hoveredNode.stage]}</div>
                <div className="mt-1" style={{ color: "#2a4a6a", fontSize: 10 }}>
                  <span style={{ color: "#4a5568" }}>IN:</span> {hoveredNode.inputs.input_type} [{hoveredNode.inputs.expected_state}]
                </div>
                <div style={{ color: "#2a4a6a", fontSize: 10 }}>
                  <span style={{ color: "#4a5568" }}>OUT:</span> {hoveredNode.outputs.output_type} [{hoveredNode.outputs.resulting_state}]
                </div>
                {hoveredNode.keywords?.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {hoveredNode.keywords.slice(0,4).map(k => (
                      <span key={k} className="px-1 rounded" style={{ background: "#0d1f3c", color: "#4a5568", fontSize: 9 }}>{k}</span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Prompt bar ── */}
          <div className="px-4 py-3 border-t" style={{ borderColor: "#0d1f3c", background: "#040a14" }}>
            <div className="flex items-center gap-3 rounded px-4 py-3" style={{ background: "#060e1c", border: "1px solid #0d1f3c" }}>
              <Terminal size={14} style={{ color: isRunning ? "#00e5ff" : "#2a4a6a" }} className={isRunning ? "animate-pulse" : ""} />
              <input
                ref={textareaRef}
                className="flex-1 bg-transparent outline-none text-sm"
                style={{ color: "#8892b0", fontSize: 12, fontFamily: "inherit" }}
                placeholder='e.g. "load sales.csv and clean nulls then group by region and export"'
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isRunning}
              />
              {prompt && !isRunning && (
                <button onClick={() => setPrompt("")} className="p-1 rounded hover:opacity-70">
                  <X size={12} style={{ color: "#2a4a6a" }} />
                </button>
              )}
              <button
                onClick={handleRun}
                disabled={isRunning || !prompt.trim()}
                className="flex items-center gap-2 px-4 py-1.5 rounded text-xs font-bold transition-all hover:opacity-90 disabled:opacity-40"
                style={{ background: isRunning ? "#0a1a2e" : "#001a2e", color: "#00e5ff", border: "1px solid #00e5ff33", fontSize: 11 }}>
                {isRunning ? <Activity size={12} className="animate-pulse" /> : <Play size={12} />}
                {isRunning ? "ROUTING..." : "RUN"}
              </button>
            </div>
            <div className="mt-1.5 flex gap-2 flex-wrap">
              {["load csv and clean nulls", "full ETL pipeline", "ML pipeline", "load json and filter rows and export html"].map(ex => (
                <button key={ex} onClick={() => setPrompt(ex)}
                  className="text-xs px-2 py-0.5 rounded hover:opacity-80 transition-all"
                  style={{ background: "#060e1c", color: "#2a4a6a", border: "1px solid #0d1f3c", fontSize: 9 }}>
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {/* ── Bottom tabs — Log / Path ── */}
          <div className="border-t" style={{ borderColor: "#0d1f3c", background: "#030712", height: 160 }}>
            <div className="flex border-b" style={{ borderColor: "#0d1f3c" }}>
              {[{ id: "log", icon: <Terminal size={11}/>, label: "Router Log" }, { id: "path", icon: <GitBranch size={11}/>, label: `Execution Path${activePath.length ? ` (${activePath.length})` : ""}` }].map(t => (
                <button key={t.id} onClick={() => setBottomTab(t.id)}
                  className="flex items-center gap-1.5 px-4 py-1.5 text-xs transition-all"
                  style={{ color: bottomTab === t.id ? "#00e5ff" : "#2a4a6a", borderBottom: bottomTab === t.id ? "1px solid #00e5ff" : "1px solid transparent", fontSize: 10 }}>
                  {t.icon}{t.label}
                </button>
              ))}
            </div>
            <div className="overflow-y-auto px-4 py-2" style={{ height: 130, scrollbarWidth: "thin", scrollbarColor: "#0d1f3c transparent" }}>
              {bottomTab === "log" ? (
                <>
                  {logs.map((e, i) => <LogEntry key={i} entry={e} />)}
                  <div ref={logEndRef} />
                </>
              ) : activePath.length === 0 ? (
                <div className="text-xs pt-4" style={{ color: "#1a2a3a", fontSize: 11 }}>No execution path yet. Run a prompt to see the routing.</div>
              ) : (
                <div className="flex flex-col gap-1 pt-1">
                  {activePath.map((cell, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-xs" style={{ color: "#2a4a6a", minWidth: 24, fontSize: 10 }}>{i+1}.</span>
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: virtualEdges.has(cell.cell_id) ? "#ff00aa" : "#00e5ff", flexShrink: 0 }} />
                      <span className="text-xs font-bold" style={{ color: virtualEdges.has(cell.cell_id) ? "#ff00aa" : "#00e5ff", fontSize: 11 }}>{cell.cell_id}</span>
                      <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>S{cell.stage} · {cell.inputs.input_type}[{cell.inputs.expected_state}] → {cell.outputs.output_type}[{cell.outputs.resulting_state}]</span>
                      {virtualEdges.has(cell.cell_id) && <span className="text-xs px-1.5 rounded" style={{ background: "#1a0620", color: "#ff00aa", fontSize: 9, border: "1px solid #ff00aa33" }}>TUNNEL</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── RIGHT PANEL — Code / Inspect ── */}
        <div className="flex flex-col border-l" style={{ borderColor: "#0d1f3c", background: "#040a14", width: 340, minWidth: 280 }}>
          <div className="flex border-b" style={{ borderColor: "#0d1f3c" }}>
            {[{ id: "code", icon: <Code2 size={11}/>, label: "Generated Code" }, { id: "inspect", icon: <Eye size={11}/>, label: selectedNode ? selectedNode.cell_id : "Inspect" }].map(t => (
              <button key={t.id} onClick={() => setRightTab(t.id)}
                className="flex items-center gap-1.5 px-4 py-2 text-xs transition-all flex-1 justify-center"
                style={{ color: rightTab === t.id ? "#00e5ff" : "#2a4a6a", borderBottom: rightTab === t.id ? "1px solid #00e5ff" : "1px solid transparent", fontSize: 10 }}>
                {t.icon}<span className="truncate">{t.label}</span>
              </button>
            ))}
          </div>

          {rightTab === "code" ? (
            <>
              <div className="flex items-center justify-between px-3 py-1.5 border-b" style={{ borderColor: "#0d1f3c" }}>
                <div className="flex items-center gap-2">
                  <FileCode size={11} style={{ color: "#4a5568" }} />
                  <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>output.py · {codeLines.length} lines</span>
                </div>
                <div className="flex gap-1">
                  <button onClick={handleCopyCode} className="flex items-center gap-1 px-2 py-1 rounded text-xs hover:opacity-80"
                    style={{ background: "#060e1c", color: copied ? "#98c379" : "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}>
                    {copied ? <CheckCircle size={10}/> : <Copy size={10}/>}
                    {copied ? "Copied!" : "Copy"}
                  </button>
                  <button className="flex items-center gap-1 px-2 py-1 rounded text-xs hover:opacity-80"
                    style={{ background: "#060e1c", color: "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}
                    title="Export (future feature)">
                    <Download size={10}/> Save
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-3" style={{ scrollbarWidth: "thin", scrollbarColor: "#0d1f3c transparent" }}>
                <div className="font-mono text-xs leading-relaxed" style={{ fontSize: 11 }}>
                  {codeLines.map((line, i) => (
                    <SyntaxLine key={i} line={line} />
                  ))}
                </div>
              </div>
            </>
          ) : selectedNode ? (
            <div className="flex-1 overflow-y-auto p-4" style={{ scrollbarWidth: "thin", scrollbarColor: "#0d1f3c transparent" }}>
              {/* Cell header */}
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: activeIds.has(selectedNode.cell_id) ? "#00e5ff" : virtualEdges.has(selectedNode.cell_id) ? "#ff00aa" : "#192850", boxShadow: activeIds.has(selectedNode.cell_id) ? "0 0 8px #00e5ff" : "none" }} />
                  <span className="font-bold text-sm" style={{ color: activeIds.has(selectedNode.cell_id) ? "#00e5ff" : "#8892b0" }}>{selectedNode.cell_id}</span>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <span className="px-2 py-0.5 rounded text-xs" style={{ background: "#060e1c", color: "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}>Stage {selectedNode.stage}: {STAGE_NAMES[selectedNode.stage]}</span>
                  {activeIds.has(selectedNode.cell_id) && <span className="px-2 py-0.5 rounded text-xs" style={{ background: "#001a2e", color: "#00e5ff", border: "1px solid #00e5ff33", fontSize: 10 }}>ACTIVE</span>}
                  {virtualEdges.has(selectedNode.cell_id) && <span className="px-2 py-0.5 rounded text-xs" style={{ background: "#1a0620", color: "#ff00aa", border: "1px solid #ff00aa33", fontSize: 10 }}>TUNNEL</span>}
                </div>
              </div>

              {/* I/O section */}
              <div className="mb-4">
                <div className="text-xs font-bold mb-2 tracking-widest" style={{ color: "#2a4a6a", fontSize: 10 }}>TYPE SIGNATURE</div>
                <div className="rounded p-3" style={{ background: "#060e1c", border: "1px solid #0d1f3c" }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10, minWidth: 24 }}>IN</span>
                    <span className="font-mono text-xs px-2 py-0.5 rounded" style={{ background: "#0a1a2e", color: "#61afef", fontSize: 10 }}>{selectedNode.inputs.input_type}</span>
                    <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>[{selectedNode.inputs.expected_state}]</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10, minWidth: 24 }}>OUT</span>
                    <span className="font-mono text-xs px-2 py-0.5 rounded" style={{ background: "#0a1a2e", color: "#98c379", fontSize: 10 }}>{selectedNode.outputs.output_type}</span>
                    <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>[{selectedNode.outputs.resulting_state}]</span>
                  </div>
                </div>
              </div>

              {/* Keywords */}
              <div className="mb-4">
                <div className="text-xs font-bold mb-2 tracking-widest" style={{ color: "#2a4a6a", fontSize: 10 }}>INTENT KEYWORDS</div>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.keywords.map(k => (
                    <span key={k} className="px-2 py-0.5 rounded text-xs" style={{ background: "#060e1c", color: "#4a5568", border: "1px solid #0d1f3c", fontSize: 10 }}>{k}</span>
                  ))}
                </div>
              </div>

              {/* Code template */}
              <div className="mb-4">
                <div className="text-xs font-bold mb-2 tracking-widest" style={{ color: "#2a4a6a", fontSize: 10 }}>CODE TEMPLATE</div>
                <div className="rounded p-3 font-mono overflow-x-auto" style={{ background: "#060e1c", border: "1px solid #0d1f3c", fontSize: 10 }}>
                  {(CODE_TEMPLATES[selectedNode.cell_id] || "# No template").split("\n").map((line, i) => (
                    <SyntaxLine key={i} line={line} />
                  ))}
                </div>
              </div>

              {/* Connections */}
              <div>
                <div className="text-xs font-bold mb-2 tracking-widest" style={{ color: "#2a4a6a", fontSize: 10 }}>TOPOLOGY CONNECTIONS</div>
                <div className="space-y-1">
                  {MOCK_CELLS.filter(c => c.type === "micro" && c.inputs.input_type === selectedNode.outputs.output_type && c.inputs.expected_state === selectedNode.outputs.resulting_state).map(c => (
                    <button key={c.cell_id} onClick={() => setSelectedNode(c)}
                      className="w-full flex items-center gap-2 rounded px-2 py-1.5 hover:opacity-80 transition-all text-left"
                      style={{ background: "#060e1c", border: "1px solid #0d1f3c" }}>
                      <ChevronRight size={10} style={{ color: "#2a4a6a" }} />
                      <span className="text-xs" style={{ color: "#4a5568", fontSize: 10 }}>{c.cell_id}</span>
                      <span className="ml-auto text-xs" style={{ color: "#1a2a3a", fontSize: 9 }}>S{c.stage}</span>
                    </button>
                  ))}
                  {MOCK_CELLS.filter(c => c.type === "micro" && c.outputs.output_type === selectedNode.inputs.input_type && c.outputs.resulting_state === selectedNode.inputs.expected_state).map(c => (
                    <button key={c.cell_id} onClick={() => setSelectedNode(c)}
                      className="w-full flex items-center gap-2 rounded px-2 py-1.5 hover:opacity-80 transition-all text-left"
                      style={{ background: "#060e1c", border: "1px solid #0a1428" }}>
                      <ChevronRight size={10} style={{ color: "#1a2a3a", transform: "rotate(180deg)" }} />
                      <span className="text-xs" style={{ color: "#2a4a6a", fontSize: 10 }}>{c.cell_id}</span>
                      <span className="ml-auto text-xs" style={{ color: "#1a2a3a", fontSize: 9 }}>S{c.stage} ←</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Eye size={24} style={{ color: "#0d1f3c", margin: "0 auto 8px" }} />
                <div className="text-xs" style={{ color: "#1a2a3a", fontSize: 11 }}>Click a node in the 3D view<br />or a cell in the browser</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status bar */}
      <div className="flex items-center justify-between px-4 py-1 border-t" style={{ borderColor: "#0d1f3c", background: "#020509", minHeight: 24 }}>
        <div className="flex items-center gap-4">
          <span className="text-xs" style={{ color: "#1a2a3a", fontSize: 9 }}>NSTL v2.1 · SOTA PERSISTENT NEURAL INTENT INTERFACE</span>
          {isRunning && <span className="text-xs flex items-center gap-1" style={{ color: "#00e5ff", fontSize: 9 }}><Activity size={9} className="animate-pulse" /> ROUTING</span>}
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs" style={{ color: "#1a2a3a", fontSize: 9 }}>THREE.js r128 · WebGL2</span>
          <span className="text-xs" style={{ color: "#1a2a3a", fontSize: 9 }}>{logs.length} log entries</span>
        </div>
      </div>
    </div>
  );
}


