import React, { Component, useRef, useMemo, useEffect, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { useStore } from "../store";
import {
  BufferAttribute,
  BufferGeometry,
  MeshBasicMaterial,
  Object3D,
  SphereGeometry,
  Vector3,
} from "three";

// 🚀 OPTIMIZATION 1: Render ALL base edges in ONE single draw call using BufferGeometry
function BaseEdges({ edges }) {
  const geometry = useMemo(() => {
    const geo = new BufferGeometry();
    const positions = new Float32Array(edges.length * 6);
    edges.forEach(([a, b], i) => {
      positions[i * 6] = a.x;
      positions[i * 6 + 1] = a.y;
      positions[i * 6 + 2] = a.z;
      positions[i * 6 + 3] = b.x;
      positions[i * 6 + 4] = b.y;
      positions[i * 6 + 5] = b.z;
    });
    geo.setAttribute("position", new BufferAttribute(positions, 3));
    return geo;
  }, [edges]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#0d1f4a" transparent opacity={0.3} />
    </lineSegments>
  );
}

// 🚀 OPTIMIZATION 2: Dormant nodes grouped efficiently
function DormantNodes({ cells, positions, activeIds }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new Object3D(), []);
  const geometry = useMemo(() => new SphereGeometry(0.3, 8, 8), []);
  const material = useMemo(
    () =>
      new MeshBasicMaterial({
        color: "#1a2a4a",
        transparent: true,
        opacity: 0.4,
      }),
    [],
  );

  const dormantCells = useMemo(
    () => cells.filter((c) => c.type === "micro" && !activeIds.has(c.cell_id)),
    [cells, activeIds],
  );

  const prevCellIds = useRef("");

  useEffect(() => {
    if (!meshRef.current) return;

    const currentIds = dormantCells.map((c) => c.cell_id).join(",");
    if (prevCellIds.current === currentIds) return;
    prevCellIds.current = currentIds;

    const rafId = requestAnimationFrame(() => {
      dormantCells.forEach((cell, i) => {
        const pos = positions[cell.cell_id];
        if (pos) {
          dummy.position.copy(pos);
          dummy.updateMatrix();
          meshRef.current.setMatrixAt(i, dummy.matrix);
        }
      });
      meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return () => cancelAnimationFrame(rafId);
  }, [dormantCells, positions, dummy]);

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  return (
    <instancedMesh
      key={dormantCells.length}
      ref={meshRef}
      args={[geometry, material, dormantCells.length]}
    />
  );
}

// 🎥 UX 1: Smooth Camera Animation on Node Click
function CameraController() {
  const selectedNode = useStore((s) => s.selectedNode);
  const cells = useStore((s) => s.cells);
  const { camera, controls, invalidate } = useThree();
  const [targetPos, setTargetPos] = useState(null);

  useEffect(() => {
    if (selectedNode) {
      // Recompute position to fly to
      const stageCells = cells.filter(
        (c) => c.stage === selectedNode.stage && c.type === "micro",
      );
      const idx = stageCells.findIndex(
        (c) => c.cell_id === selectedNode.cell_id,
      );
      const xPos = selectedNode.stage * 14;
      let pos = new Vector3(xPos, 0, 0);

      if (stageCells.length > 1) {
        const ring = Math.floor(idx / 7);
        const slot = idx % 7;
        const ringN = Math.min(7, stageCells.length - ring * 7);
        const angle = (slot / ringN) * 2 * Math.PI + ring * (Math.PI / 7);
        const radius = 5 + ring * 5;
        pos = new Vector3(
          xPos,
          radius * Math.cos(angle),
          radius * Math.sin(angle),
        );
      }
      setTargetPos(pos);
    }
  }, [selectedNode, cells]);

  useFrame(() => {
    if (targetPos && controls) {
      // Smoothly lerp camera and controls target
      controls.target.lerp(targetPos, 0.05);
      const idealCameraPos = targetPos
        .clone()
        .add(new Vector3(-10, 5, 10));
      camera.position.lerp(idealCameraPos, 0.05);
      invalidate(); // Tell the demand-loop to render this frame

      // Stop animating when close enough
      if (controls.target.distanceTo(targetPos) < 0.1) setTargetPos(null);
    }
  });
  return null;
}

function LatticeNodes() {
  const cells = useStore((s) => s.cells);
  const activePath = useStore((s) => s.activePath);
  const virtualEdges = useStore((s) => s.virtualEdges);
  const setHoveredNode = useStore((s) => s.setHoveredNode);
  const setSelectedNode = useStore((s) => s.setSelectedNode);
  const setRightActiveTab = useStore((s) => s.setRightActiveTab);
  const selectedNode = useStore((s) => s.selectedNode);
  const { invalidate } = useThree();

  // Position calculation (runs once)
  const positions = useMemo(() => {
    const pos = {};
    const stageGroups = {};
    cells.forEach((c) => {
      if (c.type === "micro") {
        stageGroups[c.stage] = stageGroups[c.stage] || [];
        stageGroups[c.stage].push(c);
      }
    });

    Object.entries(stageGroups).forEach(([stage, stageCells]) => {
      const xPos = parseInt(stage) * 14;
      stageCells.forEach((cell, idx) => {
        if (stageCells.length === 1)
          pos[cell.cell_id] = new Vector3(xPos, 0, 0);
        else {
          const ring = Math.floor(idx / 7);
          const slot = idx % 7;
          const ringN = Math.min(7, stageCells.length - ring * 7);
          const angle = (slot / ringN) * 2 * Math.PI + ring * (Math.PI / 7);
          const radius = 5 + ring * 5;
          pos[cell.cell_id] = new Vector3(
            xPos,
            radius * Math.cos(angle),
            radius * Math.sin(angle),
          );
        }
      });
    });
    return pos;
  }, [cells]);

  const activeIds = useMemo(
    () => new Set(activePath.map((c) => c.cell_id)),
    [activePath],
  );

  // With frameloop="demand", we must notify the canvas to repaint
  // whenever data-driven state changes (new query results, cells loaded).
  useEffect(() => { invalidate(); }, [cells, activePath, selectedNode, invalidate]);

  // Base edges calculation
  const baseEdges = useMemo(() => {
    const edges = [];
    const nextStageByInput = new Map();

    cells.forEach((cell) => {
      if (cell.type !== "micro" || !positions[cell.cell_id]) return;
      const key = `${cell.stage}:${cell.inputs.type_name}:${cell.inputs.state}`;
      const group = nextStageByInput.get(key) || [];
      group.push(cell);
      nextStageByInput.set(key, group);
    });

    cells.forEach((ca) => {
      if (ca.type !== "micro") return;
      if (!positions[ca.cell_id]) return;
      const key = `${ca.stage + 1}:${ca.outputs.type_name}:${ca.outputs.state}`;
      const nextCells = nextStageByInput.get(key) || [];
      nextCells.forEach((cb) => edges.push([positions[ca.cell_id], positions[cb.cell_id]]));
    });

    return edges;
  }, [cells, positions]);

  const pathEdges = useMemo(() => {
    const edges = [];
    for (let i = 0; i < activePath.length - 1; i++) {
      const a = positions[activePath[i].cell_id];
      const b = positions[activePath[i + 1].cell_id];
      if (a && b)
        edges.push({
          start: a,
          end: b,
          tunnel: virtualEdges.has(activePath[i + 1].cell_id),
        });
    }
    return edges;
  }, [activePath, positions, virtualEdges]);

  return (
    <group>
      <BaseEdges edges={baseEdges} />

      {/* Path edges rendered uniquely to stand out */}
      {pathEdges.map((e, i) => (
        <PathEdge key={`path-${i}`} edge={e} />
      ))}

      {/* Active Path Nodes */}
      {activePath.map((cell) => {
        const pos = positions[cell.cell_id];
        if (!pos) return null;
        const isTunnel = virtualEdges.has(cell.cell_id);
        const color = isTunnel ? "#ff00aa" : "#00e5ff";
        const isSelected = selectedNode?.cell_id === cell.cell_id;

        return (
          <mesh
            key={cell.cell_id}
            position={pos}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHoveredNode(cell);
              document.body.style.cursor = "pointer";
              invalidate();
            }}
            onPointerOut={() => {
              setHoveredNode(null);
              document.body.style.cursor = "default";
              invalidate();
            }}
            onClick={(e) => {
              e.stopPropagation();
              setSelectedNode(cell);
              setRightActiveTab("inspect");
              invalidate();
            }}
          >
            <sphereGeometry args={[isSelected ? 1.0 : 0.7, 16, 16]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={isSelected ? 1.5 : 0.8}
            />

            {/* Pulsing Selection Ring */}
            {isSelected && (
              <mesh scale={1.5}>
                <torusGeometry args={[1, 0.05, 16, 32]} />
                <meshBasicMaterial color="#ffffff" />
              </mesh>
            )}

            {/* Label - Only render for active path to save DOM performance */}
            <Html
              position={[0, 1.2, 0]}
              center
              style={{ pointerEvents: "none" }}
            >
              <div
                style={{
                  background: "rgba(6,9,20,0.9)",
                  color: "#fff",
                  fontSize: "clamp(10px, 0.8vw, 12px)",
                  padding: "4px 8px",
                  borderRadius: "4px",
                  border: `1px solid ${color}`,
                  whiteSpace: "nowrap",
                }}
              >
                {cell.cell_id}
              </div>
            </Html>
          </mesh>
        );
      })}

      <DormantNodes cells={cells} positions={positions} activeIds={activeIds} />
    </group>
  );
}

// ErrorBoundary to catch WebGL context loss and shader crashes
class WebGLErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error("[NSTL WebGL Error]", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          height: "100%", color: "var(--t2)", flexDirection: "column", gap: "12px",
          background: "var(--bg1)"
        }}>
          <span style={{ fontSize: "1.2rem", fontWeight: 600 }}>⚠ 3D Render Error</span>
          <span style={{ fontSize: "0.8rem", color: "var(--t3)" }}>WebGL context may have been lost. Try refreshing.</span>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: "6px 16px", background: "var(--accent)", border: "none",
              borderRadius: "6px", color: "#fff", cursor: "pointer", fontSize: "0.8rem"
            }}
          >Retry</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Memoized path edge component to avoid recreating Float32Array on every render
function PathEdge({ edge }) {
  const positions = useMemo(() => new Float32Array([
    edge.start.x, edge.start.y, edge.start.z,
    edge.end.x, edge.end.y, edge.end.z
  ]), [edge.start, edge.end]);

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={2}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial
        color={edge.tunnel ? "#ff00aa" : "#00e5ff"}
        linewidth={2}
      />
    </line>
  );
}

export default function ThreeScene() {
  return (
    <WebGLErrorBoundary>
      <Canvas
        camera={{ position: [60, 20, 40], fov: 50, near: 0.1, far: 1000 }}
        style={{ width: "100%", height: "100%" }}
        dpr={[1, 1.5]}
        frameloop="demand"
      >
        {/* Transparent background to let CSS gradient show through */}
        <fog attach="fog" args={["#03050c", 30, 150]} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[30, 30, 20]} intensity={1.2} />

        <CameraController />
        <LatticeNodes />

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.1}
          minDistance={2}
          maxDistance={5000}
          regress
        />

        {/* Bloom tuned for minimal GPU cost while preserving glow aesthetic */}
        <EffectComposer multisampling={0}>
          <Bloom
            luminanceThreshold={0.4}
            luminanceSmoothing={0.9}
            intensity={0.8}
            mipmapBlur
          />
        </EffectComposer>
      </Canvas>
    </WebGLErrorBoundary>
  );
}
