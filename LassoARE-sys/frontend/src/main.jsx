import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Plotly from "plotly.js-dist-min";
import "./styles.css";
import { runtimeLabel, sampleActionLabel } from "./runtime.js";

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

function apiUrl(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
}

const SELECTION_COLORS = [
  "#ef4444",
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
  "#f97316",
  "#14b8a6",
];

const ENCODER_LAYER_OPTIONS = [
  [256, 128, 64],
  [256, 64],
  [64, 32],
  [64],
];

const DISCRIMINATOR_LAYER_OPTIONS = [
  [256, 64],
  [64, 32],
  [64],
];

const LAMBDA_ATTENTION_OPTIONS = [0.1, 0.2, 0.5, 1.0];
const MARKER_METHOD_OPTIONS = ["t-test", "t-test_overestim_var", "wilcoxon"];

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

function layersToKey(layers) {
  return layers.join("-");
}

function keyToLayers(key) {
  return key.split("-").map((value) => Number(value));
}

function quotePython(value) {
  if (value === null || value === undefined) {
    return "None";
  }
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => quotePython(item)).join(", ")}]`;
  }
  return String(value);
}

function indentBlock(lines) {
  return lines.map((line) => `    ${line}`).join("\n");
}

function buildLassoViewCode({ activeSelection, summary, viewOneConfig, analysisConfig }) {
  const selectedIds = activeSelection?.ids || [];
  const obsCol = viewOneConfig.colorBy || summary?.default_color_by || "annotation";
  return [
    "from backend.do_lasso import do_lasso",
    "",
    `selected_ids = ${quotePython(selectedIds)}`,
    `obs_col = ${quotePython(obsCol)}`,
    "",
    "expanded_ids = do_lasso(",
    indentBlock([
      "adata=adata,",
      "user_selected_list=selected_ids,",
      `obs_col=${quotePython(obsCol)},`,
      "vis=False,",
      `vis_key=${quotePython(obsCol)},`,
      `do_correct=${quotePython(Boolean(analysisConfig.doCorrect))},`,
    ]),
    ")",
  ].join("\n");
}

function buildDownsampleCode({ summary, viewOneConfig, analysisConfig }) {
  const embeddingKey = viewOneConfig.embeddingKey || summary?.default_embedding || "X_umap";
  const colorBy = viewOneConfig.colorBy || summary?.default_color_by || "leiden";
  return [
    "from backend.do_downsample import do_h5ad_downsample",
    "",
    "downsampled_adata, nearest_ids = do_h5ad_downsample(",
    indentBlock([
      "adata=adata,",
      `sample_rate=${quotePython(Number(analysisConfig.sampleRate))},`,
      `leiden_r=${quotePython(Number(analysisConfig.leidenResolution))},`,
      `uniform_rate=${quotePython(Number(analysisConfig.uniformRate))},`,
      "add_col='orig_idx',",
      `cluster_key=${quotePython(colorBy)},`,
      `obsm_key=${quotePython(embeddingKey)},`,
    ]),
    ")",
  ].join("\n");
}

function buildLassoARECode({ confirmedSelections, analysisConfig }) {
  const userSelectedLists = confirmedSelections.map((selection) => selection.ids);
  const selectionLabels = confirmedSelections.map((selection) => selection.displayName || selection.variableName);
  const encLayers = keyToLayers(analysisConfig.encoderLayersKey);
  const decLayers = encLayers.slice().reverse();
  const discLayers = keyToLayers(analysisConfig.discriminatorLayersKey);
  const nClusters = analysisConfig.nClusters.trim() ? Number(analysisConfig.nClusters.trim()) : null;
  const pretrainEpoch = Number(analysisConfig.pretrainEpoch);
  const mode = analysisConfig.lassoareMode;
  const commonArgs = [
    "adata=adata,",
    `user_selected_lists=${quotePython(userSelectedLists)},`,
    `n_clusters=${quotePython(nClusters)},`,
    `enc_pretrain_epoch=${quotePython(pretrainEpoch)},`,
    `disc_pretrain_epoch=${quotePython(pretrainEpoch)},`,
    `gan_epoch=${quotePython(Number(analysisConfig.trainingEpoch))},`,
    `enc_layers=${quotePython(encLayers)},`,
    `dec_layers=${quotePython(decLayers)},`,
    `disc_layers=${quotePython(discLayers)},`,
    "batch_size=256,",
    `lambda_attention=${quotePython(Number(analysisConfig.lambdaAttention))},`,
    `leiden_r=${quotePython(Number(analysisConfig.leidenResolution))},`,
    "z_dim=32,",
    `is_pca=${quotePython(Boolean(analysisConfig.isPca))},`,
    "do_pp=False,",
  ];

  if (mode === "generate") {
    return [
      "from backend.LassoARE.reconstruction import reconstruction_with_lasso_are",
      "",
      `# Prior groups: ${selectionLabels.join(", ")}`,
      "",
      "result_adata = reconstruction_with_lasso_are(",
      indentBlock([
        ...commonArgs.slice(0, 2),
        "using_emb=None,",
        ...commonArgs.slice(2),
      ]),
      ")",
    ].join("\n");
  }

  return [
    "from backend.LassoARE.reconstruction import reconstruction_with_ref",
    "",
    `# Prior groups: ${selectionLabels.join(", ")}`,
    "",
    "result_adata = reconstruction_with_ref(",
    indentBlock([
      ...commonArgs,
      `using_emb=${quotePython(analysisConfig.reconstructEmbeddingKey)},`,
      `ref_enc_layers=${quotePython(encLayers)},`,
      `ref_pretrain_epoch=${quotePython(pretrainEpoch)},`,
      `lambda_ref=${quotePython(Number(analysisConfig.lambdaRef))},`,
    ]),
    ")",
  ].join("\n");
}

function createViewConfig(summary, plot, overrides = {}) {
  return {
    embeddingKey: plot?.embedding_key || summary?.default_embedding || "",
    colorBy: plot?.color_by || summary?.default_color_by || "",
    pointSize: 4.5,
    opacity: 0.82,
    invertX: false,
    invertY: false,
    ...overrides,
  };
}

function makeCategoryColors(values) {
  const fallback = "#90a4b8";
  if (!values) {
    return { colors: [], legend: [] };
  }

  const categories = [];
  const seen = new Set();
  values.forEach((value) => {
    const key = value === null || value === undefined ? "Unassigned" : String(value);
    if (!seen.has(key)) {
      seen.add(key);
      categories.push(key);
    }
  });

  const palette = categories.map((_, index) => {
    const hue = Math.round((index * 137.508) % 360);
    return `hsl(${hue}, 68%, 52%)`;
  });
  const lookup = new Map(categories.map((item, index) => [item, palette[index] || fallback]));

  return {
    colors: values.map((value) => lookup.get(value === null || value === undefined ? "Unassigned" : String(value)) || fallback),
    legend: categories.map((label) => ({ label, color: lookup.get(label) || fallback })),
  };
}


function markerGeneColor(gene, index) {
  const hue = Math.round((index * 137.508) % 360);
  return `hsl(${hue}, 70%, 46%)`;
}

function markerPointSize(fraction) {
  const bounded = Math.max(0, Math.min(1, Number(fraction) || 0));
  if (bounded <= 0) {
    return 3;
  }
  return 5 + Math.sqrt(bounded) * 24;
}

function formatMarkerNumber(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}


function arrayUnion(left, right) {
  const next = new Set(left);
  right.forEach((item) => next.add(item));
  return Array.from(next).sort((a, b) => a - b);
}

function selectionKindConfig(kind) {
  if (kind === "refining") {
    return {
      displayLabelPrefix: "Refining",
      variablePrefix: "refine_list",
    };
  }
  return {
    displayLabelPrefix: "Selection",
    variablePrefix: "select_list",
  };
}

function normalizeSelectionCatalog(selections) {
  const counters = new Map();
  return selections.map((selection, index) => {
    const kind = selection.kind || "selection";
    const count = (counters.get(kind) || 0) + 1;
    counters.set(kind, count);
    const kindMeta = selectionKindConfig(kind);
    return {
      ...selection,
      kind,
      displayName: `${kindMeta.displayLabelPrefix} ${count}`,
      variableName: `${kindMeta.variablePrefix}${count}`,
      color: selection.color || SELECTION_COLORS[index % SELECTION_COLORS.length],
    };
  });
}

function buildBaseColorArray(plotData, baseColors, selectedOnly) {
  const fallback = Array(plotData.points.ids.length).fill("#7aa3f0");
  const source = baseColors.length ? baseColors : fallback;
  return selectedOnly ? Array(plotData.points.ids.length).fill("#d8e0ea") : source;
}

function makePlotColorInfo(plotData) {
  if (plotData?.color_mode === "continuous") {
    return {
      colors: plotData.points?.expression_values || [],
      legend: [],
      continuous: true,
    };
  }
  return {
    ...makeCategoryColors(plotData?.points?.color_values),
    continuous: false,
  };
}

function buildSubsetTrace(plotData, ids, color, pointSize, opacity, lineColor, lineWidth) {
  const xs = [];
  const ys = [];
  const customdata = [];
  ids.forEach((id) => {
    xs.push(plotData.points.x[id]);
    ys.push(plotData.points.y[id]);
    customdata.push(id);
  });

  return {
    x: xs,
    y: ys,
    type: "scattergl",
    mode: "markers",
    hovertemplate: "Cell ID: %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
    customdata,
    marker: {
      size: pointSize,
      opacity,
      color,
      line: {
        color: lineColor,
        width: lineWidth,
      },
    },
  };
}

function buildPlotTraces({
  plotData,
  colorInfo,
  viewConfig,
  interactive,
  selectedOnly,
  pendingIds,
  confirmedSelections,
}) {
  const expressionValues = plotData.points.expression_values || [];
  const useContinuousColor = colorInfo.continuous && !(!interactive && selectedOnly);
  const marker = {
    size: viewConfig.pointSize,
    opacity: viewConfig.opacity,
    color: buildBaseColorArray(plotData, colorInfo.colors, !interactive && selectedOnly),
    line: {
      color: "rgba(255,255,255,0.7)",
      width: 0.5,
    },
  };
  if (useContinuousColor) {
    marker.colorscale = "Viridis";
    marker.cmin = plotData.expression_min ?? 0;
    marker.cmax = plotData.expression_max ?? Math.max(...expressionValues, 1);
    marker.colorbar = {
      title: { text: plotData.color_label || "Expression" },
      thickness: 12,
    };
  }

  const baseTrace = {
    x: plotData.points.x,
    y: plotData.points.y,
    type: "scattergl",
    mode: "markers",
    hovertemplate: colorInfo.continuous
      ? "Cell ID: %{customdata}<br>Expression: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
      : "Cell ID: %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
    text: colorInfo.continuous ? expressionValues.map((value) => formatMarkerNumber(value, 3)) : undefined,
    customdata: plotData.points.ids,
    marker,
  };

  const traces = [baseTrace];

  if (!interactive) {
    confirmedSelections.forEach((selection) => {
      if (selection.ids.length) {
        traces.push(buildSubsetTrace(
          plotData,
          selection.ids,
          selection.color,
          Math.max(viewConfig.pointSize + 0.4, 5),
          0.96,
          "#ffffff",
          0.8,
        ));
      }
    });
  }

  if (pendingIds.length) {
    traces.push(buildSubsetTrace(
      plotData,
      pendingIds,
      interactive ? "#0f172a" : "#111827",
      Math.max(viewConfig.pointSize + 0.8, 5.4),
      0.98,
      "#ffffff",
      1.1,
    ));
  }

  return traces;
}


function getPaddedRange(values) {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (!finiteValues.length) {
    return [-1, 1];
  }
  const min = Math.min(...finiteValues);
  const max = Math.max(...finiteValues);
  const span = Math.max(max - min, 1);
  const pad = span * 0.06;
  return [min - pad, max + pad];
}

function buildInitialRanges(plotData, viewConfig) {
  const xRange = getPaddedRange(plotData.points.x);
  const yRange = getPaddedRange(plotData.points.y);
  return {
    x: viewConfig.invertX ? [xRange[1], xRange[0]] : xRange,
    y: viewConfig.invertY ? [yRange[1], yRange[0]] : yRange,
  };
}

function zoomRangeAroundCenter(currentRange, baseRange, zoomLevel) {
  const center = (Number(currentRange[0]) + Number(currentRange[1])) / 2;
  const width = Math.abs(baseRange[1] - baseRange[0]) / zoomLevel;
  const reversed = currentRange[0] > currentRange[1];
  return reversed ? [center + width / 2, center - width / 2] : [center - width / 2, center + width / 2];
}

function rangeFromRelayout(event, axisName, fallbackRange) {
  const fullRange = event?.[`${axisName}.range`];
  if (Array.isArray(fullRange) && fullRange.length >= 2) {
    return [Number(fullRange[0]), Number(fullRange[1])];
  }
  const start = event?.[`${axisName}.range[0]`];
  const end = event?.[`${axisName}.range[1]`];
  if (start !== undefined && end !== undefined) {
    return [Number(start), Number(end)];
  }
  return fallbackRange;
}


function idsFromPlotlySelection(event) {
  return event?.points
    ? event.points
      .filter((point) => point.curveNumber === 0)
      .map((point) => Number(point.customdata ?? point.pointIndex))
      .filter((id) => Number.isFinite(id))
    : [];
}

function removePlotlySelectionShapes(graphDiv) {
  graphDiv.querySelectorAll(".selectionlayer > *, .select-outline, .selection-outline").forEach((node) => node.remove());
}

function clearPlotlySelection(graphDiv) {
  window.requestAnimationFrame(() => {
    Plotly.restyle(graphDiv, { selectedpoints: [null] });
    Plotly.relayout(graphDiv, { selections: [] });
    removePlotlySelectionShapes(graphDiv);
    window.setTimeout(() => removePlotlySelectionShapes(graphDiv), 0);
  });
}

function SelectionTextModal({ open, title, text, copied, onClose, onCopy }) {
  if (!open) {
    return null;
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{title}</h3>
            <p>Copy the selected IDs in a Python-friendly format.</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>Close</button>
        </div>
        <textarea readOnly value={text} className="id-textarea" />
        <div className="modal-actions">
          <button type="button" onClick={onCopy}>Copy content</button>
          <button type="button" className="ghost-button" onClick={onClose}>Dismiss</button>
          {copied ? <span className="copy-hint">Copied to clipboard.</span> : null}
        </div>
      </div>
    </div>
  );
}

function SelectionTabs({
  selections,
  activeSelectionId,
  overlapCountMap,
  onSelect,
  onShowIds,
  onDelete,
}) {
  if (!selections.length) {
    return (
      <section className="selection-tabs panel-card">
        <div className="section-heading">
          <h3>Selection classes</h3>
          <p>No confirmed selection classes yet. Confirm a draft from View 1 to create one.</p>
        </div>
      </section>
    );
  }

  const activeSelection = selections.find((selection) => selection.id === activeSelectionId) || selections[selections.length - 1];

  return (
    <section className="selection-tabs panel-card">
      <div className="section-heading">
        <h3>Selection classes</h3>
        <p>Review each confirmed selection class, inspect its IDs, or remove it.</p>
      </div>
      <div className="selection-tab-row">
        {selections.map((selection, index) => (
          <button
            type="button"
            key={selection.id}
            className={classNames("selection-tab", activeSelection.id === selection.id && "selection-tab-active")}
            onClick={() => onSelect(selection.id)}
          >
            <span className="selection-tab-dot" style={{ backgroundColor: selection.color }} />
            <span>{selection.displayName || `Selection ${index + 1}`}</span>
          </button>
        ))}
      </div>
      <div className="selection-tab-panel">
        <div className="selection-tab-meta">
          <div>
            <span className="summary-label">Python variable</span>
            <strong>{activeSelection.variableName}</strong>
          </div>
          <div>
            <span className="summary-label">Cells in this class</span>
            <strong>{activeSelection.ids.length.toLocaleString()}</strong>
          </div>
          <div>
            <span className="summary-label">Overlaps in this class</span>
            <strong>{(overlapCountMap.get(activeSelection.id) || 0).toLocaleString()}</strong>
          </div>
        </div>
        <div className="selection-tab-actions">
          <button type="button" className="ghost-button" onClick={() => onShowIds(activeSelection)}>
            Show IDs
          </button>
          <button type="button" className="ghost-button danger-button" onClick={() => onDelete(activeSelection.id)}>
            Delete this selection
          </button>
        </div>
      </div>
    </section>
  );
}

function SettingsPanel({
  title,
  open,
  onToggle,
  summary,
  config,
  onConfigChange,
  busy,
  geneExpressionValue = "",
  onGeneExpressionChange = null,
  onShowGeneExpression = null,
}) {
  return (
    <section className="side-card side-card-settings">
      <button type="button" className="section-toggle" onClick={onToggle}>
        <span>{title}</span>
        <span>{open ? "Hide" : "Open"}</span>
      </button>
      {open ? (
        <div className="settings-stack">
          <label>
            <span>Embedding</span>
            <select
              value={config.embeddingKey}
              onChange={(event) => onConfigChange({ embeddingKey: event.target.value })}
              disabled={!summary?.available_embeddings?.length || busy}
            >
              <option value="">Select an embedding</option>
              {(summary?.available_embeddings || []).map((key) => (
                <option value={key} key={key}>{key}</option>
              ))}
            </select>
          </label>

          <label>
            <span>Color by</span>
            <select
              value={config.colorBy}
              onChange={(event) => onConfigChange({ colorBy: event.target.value })}
              disabled={!summary?.obs_columns?.length || busy}
            >
              <option value="">No coloring</option>
              {(summary?.obs_columns || []).map((column) => (
                <option value={column} key={column}>{column}</option>
              ))}
            </select>
          </label>

          <label className="slider-block">
            <div className="slider-label">
              <span>Cell size</span>
              <strong>{config.pointSize.toFixed(1)}</strong>
            </div>
            <input
              type="range"
              min="2"
              max="16"
              step="0.5"
              value={config.pointSize}
              onChange={(event) => onConfigChange({ pointSize: Number(event.target.value) })}
            />
          </label>

          <label className="slider-block">
            <div className="slider-label">
              <span>Opacity</span>
              <strong>{config.opacity.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0.15"
              max="1"
              step="0.05"
              value={config.opacity}
              onChange={(event) => onConfigChange({ opacity: Number(event.target.value) })}
            />
          </label>

          <div className="check-grid">
            <label className="checkbox-line">
              <input
                type="checkbox"
                checked={config.invertX}
                onChange={(event) => onConfigChange({ invertX: event.target.checked })}
              />
              <span>Inverse X axis</span>
            </label>
            <label className="checkbox-line">
              <input
                type="checkbox"
                checked={config.invertY}
                onChange={(event) => onConfigChange({ invertY: event.target.checked })}
              />
              <span>Inverse Y axis</span>
            </label>
          </div>

          {onShowGeneExpression ? (
            <div className="gene-expression-tool">
              <label>
                <span>Gene expression</span>
                <input
                  type="text"
                  value={geneExpressionValue}
                  placeholder="IL2RA"
                  onChange={(event) => onGeneExpressionChange?.(event.target.value)}
                  disabled={busy || !summary}
                />
              </label>
              <button
                type="button"
                className="ghost-button"
                onClick={onShowGeneExpression}
                disabled={busy || !summary || !geneExpressionValue.trim()}
              >
                Show In Fig2
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function PlotPanel({
  title,
  subtitle,
  plotData,
  viewConfig,
  interactive,
  selectedOnly,
  pendingIds,
  confirmedSelections,
  onAddSelectionIds,
  onCancelPending,
  onConfirmPending,
  confirmButtonLabel = "Confirm",
  draftNote = "Each completed box or lasso action adds to the draft selection.",
  statusHint = null,
}) {
  const plotRef = useRef(null);
  const graphRef = useRef(null);
  const [dragMode, setDragMode] = useState("pan");
  const [legendExpanded, setLegendExpanded] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const currentRangesRef = useRef(null);
  const selectingIdsRef = useRef([]);
  const lastCommittedSelectionRef = useRef({ key: "", time: 0 });
  const colorInfo = useMemo(() => makePlotColorInfo(plotData), [plotData]);
  const baseRanges = useMemo(() => (plotData ? buildInitialRanges(plotData, viewConfig) : null), [plotData, viewConfig.invertX, viewConfig.invertY]);

  useEffect(() => {
    currentRangesRef.current = baseRanges;
    setZoomLevel(1);
  }, [baseRanges]);

  useEffect(() => {
    if (!baseRanges || !graphRef.current) {
      return;
    }
    const currentRanges = currentRangesRef.current || baseRanges;
    const nextRanges = {
      x: zoomRangeAroundCenter(currentRanges.x, baseRanges.x, zoomLevel),
      y: zoomRangeAroundCenter(currentRanges.y, baseRanges.y, zoomLevel),
    };
    currentRangesRef.current = nextRanges;
    Plotly.relayout(graphRef.current, {
      "xaxis.range": nextRanges.x,
      "yaxis.range": nextRanges.y,
    });
  }, [zoomLevel, baseRanges]);

  useEffect(() => {
    if (!plotData || !plotRef.current) {
      return;
    }

    const graphDiv = plotRef.current;
    graphRef.current = graphDiv;

    const traces = buildPlotTraces({
      plotData,
      colorInfo,
      viewConfig,
      interactive,
      selectedOnly,
      pendingIds,
      confirmedSelections,
    });

    const visibleRanges = currentRangesRef.current || baseRanges;

    const layout = {
      dragmode: interactive ? dragMode : "pan",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.72)",
      margin: { l: 58, r: colorInfo.continuous ? 86 : 20, t: 18, b: 52 },
      xaxis: {
        title: plotData.x_label,
        zeroline: false,
        gridcolor: "#dfe7f0",
        linecolor: "#c7d4e2",
        range: visibleRanges?.x,
        autorange: false,
      },
      yaxis: {
        title: plotData.y_label,
        zeroline: false,
        gridcolor: "#dfe7f0",
        linecolor: "#c7d4e2",
        range: visibleRanges?.y,
        autorange: false,
      },
      font: {
        family: "IBM Plex Sans, sans-serif",
        color: "#29415a",
      },
      showlegend: false,
    };

    const config = {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      scrollZoom: false,
      doubleClick: false,
      showTips: false,
    };

    Plotly.react(graphDiv, traces, layout, config);
    graphDiv.removeAllListeners?.("plotly_relayout");
    graphDiv.on("plotly_relayout", (event) => {
      const previousRanges = currentRangesRef.current || baseRanges;
      currentRangesRef.current = {
        x: rangeFromRelayout(event, "xaxis", previousRanges.x),
        y: rangeFromRelayout(event, "yaxis", previousRanges.y),
      };
    });

    const commitSelectionIds = (ids) => {
      const uniqueIds = Array.from(new Set(ids)).sort((a, b) => a - b);
      if (!uniqueIds.length) {
        return;
      }
      const now = Date.now();
      const key = uniqueIds.join(",");
      const last = lastCommittedSelectionRef.current;
      if (last.key === key && now - last.time < 300) {
        clearPlotlySelection(graphDiv);
        return;
      }
      lastCommittedSelectionRef.current = { key, time: now };
      selectingIdsRef.current = [];
      clearPlotlySelection(graphDiv);
      window.requestAnimationFrame(() => onAddSelectionIds(uniqueIds));
    };

    if (interactive) {
      graphDiv.removeAllListeners?.("plotly_selecting");
      graphDiv.removeAllListeners?.("plotly_selected");
      graphDiv.on("plotly_selecting", (event) => {
        selectingIdsRef.current = idsFromPlotlySelection(event);
      });
      graphDiv.on("plotly_selected", (event) => {
        commitSelectionIds(idsFromPlotlySelection(event));
      });
    }

    const handlePointerUp = () => {
      if (!interactive || !["select", "lasso"].includes(dragMode)) {
        return;
      }
      const ids = selectingIdsRef.current;
      if (ids.length) {
        window.setTimeout(() => commitSelectionIds(ids), 0);
      } else {
        window.setTimeout(() => clearPlotlySelection(graphDiv), 0);
      }
    };
    graphDiv.addEventListener("mouseup", handlePointerUp);
    graphDiv.addEventListener("touchend", handlePointerUp);

    return () => {
      graphDiv.removeEventListener("mouseup", handlePointerUp);
      graphDiv.removeEventListener("touchend", handlePointerUp);
    };
  }, [plotData, viewConfig, interactive, selectedOnly, pendingIds, confirmedSelections, colorInfo, dragMode, onAddSelectionIds, baseRanges]);

  useEffect(() => {
    return () => {
      if (graphRef.current) {
        Plotly.purge(graphRef.current);
      }
    };
  }, []);

  const legendItems = colorInfo.continuous ? [] : (legendExpanded ? colorInfo.legend : colorInfo.legend.slice(0, 18));

  if (!plotData) {
    return (
      <section className="plot-shell panel-card empty-plot">
        <p>No embedding is displayed yet.</p>
        <span>Upload a dataset or choose a sample file from the right sidebar.</span>
      </section>
    );
  }

  return (
    <section className={classNames("plot-shell", "panel-card", !interactive && "plot-shell-readonly")}>
      <div className="plot-header">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <div className="toolbar-meta">
          <span>{plotData.color_mode === "continuous" ? plotData.color_label : plotData.embedding_key}</span>
          <span>{interactive ? `${pendingIds.length} pending` : `${confirmedSelections.length} confirmed`}</span>
        </div>
      </div>

      {interactive ? (
        <div className="selection-action-bar">
          <div className="plot-toolbar">
            <div className="toolbar-group">
              <button type="button" className={classNames("ghost-button", dragMode === "select" && "active")} onClick={() => setDragMode("select")}>
                Box Select
              </button>
              <button type="button" className={classNames("ghost-button", dragMode === "lasso" && "active")} onClick={() => setDragMode("lasso")}>
                Lasso Tool
              </button>
              <button type="button" className={classNames("ghost-button", dragMode === "pan" && "active")} onClick={() => setDragMode("pan")}>
                Pan
              </button>
            </div>
            <div className="toolbar-group">
              <button type="button" className="ghost-button" onClick={onCancelPending} disabled={!pendingIds.length}>
                Cancel
              </button>
              <button type="button" onClick={onConfirmPending} disabled={!pendingIds.length}>
                {confirmButtonLabel}
              </button>
            </div>
          </div>
          <p className="draft-note">{draftNote}</p>
        </div>
      ) : (
        <div className="readonly-note">
          <span>{statusHint || "Preview only"}</span>
          <span>{plotData.color_mode === "continuous" ? "Expression intensity is shown with a continuous color scale." : (selectedOnly ? "Confirmed classes stay vivid while other cells turn grey." : "Confirmed classes overlay on top of the original coloring.")}</span>
        </div>
      )}

      <div className="plot-stage">
        <div ref={plotRef} className="plot-area" />
      </div>

      <div className="plot-zoom-control">
        <label htmlFor={`${title}-zoom`}>
          <span>Zoom</span>
          <strong>{zoomLevel.toFixed(1)}x</strong>
        </label>
        <input
          id={`${title}-zoom`}
          type="range"
          min="1"
          max="8"
          step="0.1"
          value={zoomLevel}
          onChange={(event) => setZoomLevel(Number(event.target.value))}
        />
      </div>

      <div className="legend-wrap">
        {legendItems.map((item) => (
          <div key={item.label} className="legend-item">
            <span className="legend-swatch" style={{ backgroundColor: item.color }} />
            <span>{item.label}</span>
          </div>
        ))}
        {colorInfo.legend.length > 18 ? (
          <button type="button" className="legend-toggle" onClick={() => setLegendExpanded((value) => !value)}>
            {legendExpanded ? "Collapse categories" : `+${colorInfo.legend.length - 18} more categories`}
          </button>
        ) : null}
      </div>
    </section>
  );
}


function MarkerBubblePlot({ markerResult }) {
  const plotRef = useRef(null);
  const geneColors = useMemo(() => {
    const lookup = new Map();
    (markerResult?.genes || []).forEach((gene, index) => {
      lookup.set(gene, markerGeneColor(gene, index));
    });
    return lookup;
  }, [markerResult]);

  useEffect(() => {
    if (!markerResult || !plotRef.current) {
      return undefined;
    }

    const traces = (markerResult.genes || []).map((gene, geneIndex) => {
      const genePoints = (markerResult.points || []).filter((point) => point.gene === gene && point.is_marker);
      return {
        name: gene,
        x: genePoints.map((point) => point.group),
        y: genePoints.map((point) => point.gene),
        type: "scatter",
        mode: "markers",
        customdata: genePoints.map((point) => ([
          formatMarkerNumber(point.mean_expression, 3),
          formatMarkerNumber(point.fraction * 100, 1),
          point.rank ? `#${point.rank}` : "not top marker",
          formatMarkerNumber(point.score, 3),
          formatMarkerNumber(point.pval_adj, 3),
          formatMarkerNumber(point.logfoldchange, 3),
        ])),
        hovertemplate: [
          "Group: %{x}",
          "Gene: %{y}",
          "Mean expression: %{customdata[0]}",
          "Cells expressed: %{customdata[1]}%",
          "Marker rank: %{customdata[2]}",
          "Score: %{customdata[3]}",
          "Adj. p-value: %{customdata[4]}",
          "Log fold change: %{customdata[5]}",
          "<extra></extra>",
        ].join("<br>"),
        marker: {
          size: genePoints.map((point) => markerPointSize(point.fraction)),
          color: geneColors.get(gene) || markerGeneColor(gene, geneIndex),
          opacity: genePoints.map((point) => (point.mean_expression > 0 ? 0.9 : 0.18)),
          line: {
            color: "rgba(20, 37, 58, 0.5)",
            width: 1.1,
          },
        },
      };
    });

    const height = Math.max(520, Math.min(980, 180 + (markerResult.genes || []).length * 22));
    const layout = {
      height,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.72)",
      margin: { l: 120, r: markerResult.genes.length <= 35 ? 170 : 28, t: 24, b: 82 },
      xaxis: {
        title: "Groups",
        type: "category",
        categoryorder: "array",
        categoryarray: markerResult.groups,
        tickangle: markerResult.groups.length > 14 ? -45 : 0,
        gridcolor: "#dfe7f0",
        linecolor: "#c7d4e2",
      },
      yaxis: {
        title: "Top marker genes",
        type: "category",
        categoryorder: "array",
        categoryarray: markerResult.genes,
        autorange: "reversed",
        gridcolor: "#edf3f8",
        linecolor: "#c7d4e2",
      },
      font: {
        family: "IBM Plex Sans, sans-serif",
        color: "#29415a",
      },
      showlegend: markerResult.genes.length <= 35,
      legend: {
        title: { text: "Marker genes" },
        x: 1.02,
        y: 1,
        bgcolor: "rgba(255,255,255,0.78)",
      },
    };

    Plotly.react(plotRef.current, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: true,
    });

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [markerResult, geneColors]);

  if (!markerResult) {
    return null;
  }

  return (
    <section className="panel-card marker-panel">
      <div className="plot-header">
        <div>
          <h3>Marker Tables</h3>
          <p>{markerResult.obs_col} | {markerResult.method} | top {markerResult.top_n} genes per group</p>
        </div>
        <div className="toolbar-meta">
          <span>{markerResult.groups.length} groups</span>
          <span>{markerResult.genes.length} marker genes</span>
        </div>
      </div>
      <div className="marker-plot-layout">
        <div className="marker-plot-frame">
          <div ref={plotRef} className="marker-plot" />
        </div>
        <aside className="marker-size-scale" aria-label="Expression fraction scale">
          <strong>Expression &gt; 0</strong>
          {[1, 0.5, 0.25].map((fraction) => (
            <div className="marker-scale-row" key={fraction}>
              <span
                className="marker-scale-dot"
                style={{
                  width: `${markerPointSize(fraction)}px`,
                  height: `${markerPointSize(fraction)}px`,
                }}
              />
              <span>{Math.round(fraction * 100)}%</span>
            </div>
          ))}
        </aside>
      </div>
      <div className="marker-scale-note">
        <span>Only top marker genes are shown</span>
        <span>Circle size: cells with expression &gt; 0</span>
        <span>Color: marker gene</span>
      </div>
    </section>
  );
}


function AnalysisPanel({
  summary,
  viewTwoSummary,
  activeSelection,
  confirmedSelections,
  analysisConfig,
  onConfigChange,
  onRunLassoView,
  onRunDownsample,
  onRunLassoare,
  markerConfig,
  onMarkerConfigChange,
  onRunMarkerPlot,
  onShowLassoViewCode,
  onShowDownsampleCode,
  onShowLassoARECode,
  onDownloadArtifact,
  job,
  disabled,
}) {
  const analysisBusy = disabled || (job && !["completed", "failed"].includes(job.status));
  const availableEmbeddings = summary?.available_embeddings || [];
  const selectedLassoViewSelection = confirmedSelections.find((selection) => selection.id === analysisConfig.lassoViewSelectionId) || activeSelection;
  const selectedLassoareIds = analysisConfig.lassoareSelectionIds || [];
  const markerSummary = markerConfig.sourceView === "view2" ? (viewTwoSummary || summary) : summary;
  const markerObsColumns = markerSummary?.obs_columns || [];
  const resolvedMarkerObsCol = markerConfig.obsCol || markerSummary?.default_color_by || markerObsColumns[0] || "";

  return (
    <section className="panel-card analysis-panel">
      <div className="section-heading">
        <h2>Analysis</h2>
        <p>Run selection-aware analysis directly under the UMAP workspace. View 1 stays on the original dataset, while View 2 shows derived results.</p>
      </div>

      <div className="analysis-stack">
        <div className="analysis-block">
          <div className="analysis-header">
            <strong>Lasso-View</strong>
            <span>{selectedLassoViewSelection ? `${selectedLassoViewSelection.ids.length} selected cells` : "No target class"}</span>
          </div>
          <p>Pick which selection class to refine, expand it with the compiled LassoView backend, and also save the propagated set back into the selection list as `Refining n`.</p>
          <label>
            <span>Selection class</span>
            <select
              value={analysisConfig.lassoViewSelectionId}
              onChange={(event) => onConfigChange({ lassoViewSelectionId: event.target.value })}
              disabled={analysisBusy || !confirmedSelections.length}
            >
              <option value="">Select one class</option>
              {confirmedSelections.map((selection) => (
                <option key={selection.id} value={selection.id}>
                  {selection.displayName || selection.variableName}
                </option>
              ))}
            </select>
          </label>
          <label className="checkbox-line">
            <input
              type="checkbox"
              checked={analysisConfig.doCorrect}
              onChange={(event) => onConfigChange({ doCorrect: event.target.checked })}
              disabled={analysisBusy}
            />
            <span>Use corrected output</span>
          </label>
          <div className="toolbar-group">
            <button type="button" onClick={onRunLassoView} disabled={analysisBusy || !selectedLassoViewSelection}>
              Run Lasso-View
            </button>
            <button type="button" className="ghost-button" onClick={onShowLassoViewCode} disabled={!selectedLassoViewSelection}>
              Show Code
            </button>
          </div>
        </div>

        <div className="analysis-block">
          <div className="analysis-header">
            <strong>Downsample</strong>
            <span>{summary?.default_embedding || "No embedding"}</span>
          </div>
          <div className="analysis-grid">
            <label>
              <span>Sample rate</span>
              <input
                type="number"
                min="0.001"
                max="1"
                step="0.01"
                value={analysisConfig.sampleRate}
                onChange={(event) => onConfigChange({ sampleRate: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>
            <label>
              <span>Uniform share</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={analysisConfig.uniformRate}
                onChange={(event) => onConfigChange({ uniformRate: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>
            <label>
              <span>Leiden resolution</span>
              <input
                type="number"
                min="0.1"
                max="5"
                step="0.1"
                value={analysisConfig.leidenResolution}
                onChange={(event) => onConfigChange({ leidenResolution: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>
          </div>
          <div className="toolbar-group">
            <button type="button" onClick={onRunDownsample} disabled={analysisBusy || !availableEmbeddings.length}>
              Run Downsample
            </button>
            <button type="button" className="ghost-button" onClick={onShowDownsampleCode} disabled={!availableEmbeddings.length}>
              Show Code
            </button>
          </div>
        </div>

        <div className="analysis-block marker-config-block">
          <div className="analysis-header">
            <strong>Marker Tables</strong>
            <span>{markerConfig.sourceView === "view2" ? "fig2" : "fig1"}</span>
          </div>
          <div className="analysis-grid">
            <label>
              <span>Figure</span>
              <select
                value={markerConfig.sourceView}
                onChange={(event) => onMarkerConfigChange({ sourceView: event.target.value, obsCol: "" })}
                disabled={analysisBusy || !summary}
              >
                <option value="view1">fig1 / View 1</option>
                <option value="view2">fig2 / View 2</option>
              </select>
            </label>
            <label>
              <span>obs column</span>
              <select
                value={resolvedMarkerObsCol}
                onChange={(event) => onMarkerConfigChange({ obsCol: event.target.value })}
                disabled={analysisBusy || !markerObsColumns.length}
              >
                <option value="">Select an obs column</option>
                {markerObsColumns.map((column) => (
                  <option key={column} value={column}>{column}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Method</span>
              <select
                value={markerConfig.method}
                onChange={(event) => onMarkerConfigChange({ method: event.target.value })}
                disabled={analysisBusy}
              >
                {MARKER_METHOD_OPTIONS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="toolbar-group">
            <button type="button" onClick={() => onRunMarkerPlot(resolvedMarkerObsCol)} disabled={analysisBusy || !markerSummary || !resolvedMarkerObsCol}>
              Generate Marker Tables
            </button>
          </div>
        </div>

        <div className="analysis-block">
          <div className="analysis-header">
            <strong>Lasso-ARE</strong>
            <span>{selectedLassoareIds.length} prior groups</span>
          </div>
          <p>Select the prior groups to use for Lasso-ARE. Any propagated LassoView result saved as `Refining n` will also appear here.</p>
          <div className="analysis-selection-list">
            {confirmedSelections.length ? confirmedSelections.map((selection) => (
              <label key={selection.id} className="checkbox-line">
                <input
                  type="checkbox"
                  checked={selectedLassoareIds.includes(selection.id)}
                  onChange={(event) => {
                    const currentIds = analysisConfig.lassoareSelectionIds || [];
                    const nextIds = event.target.checked
                      ? [...currentIds, selection.id]
                      : currentIds.filter((id) => id !== selection.id);
                    onConfigChange({ lassoareSelectionIds: nextIds });
                  }}
                  disabled={analysisBusy}
                />
                <span>{selection.displayName || selection.variableName}</span>
              </label>
            )) : <p>No selection classes available yet.</p>}
          </div>
          <div className="analysis-grid">
            <label>
              <span>Mode</span>
              <select
                value={analysisConfig.lassoareMode}
                onChange={(event) => onConfigChange({ lassoareMode: event.target.value })}
                disabled={analysisBusy}
              >
                <option value="generate">Generate</option>
                <option value="reconstruct_embedding">Reconstruct Embedding</option>
              </select>
            </label>

            <label>
              <span>Encoder layers</span>
              <select
                value={analysisConfig.encoderLayersKey}
                onChange={(event) => onConfigChange({ encoderLayersKey: event.target.value })}
                disabled={analysisBusy}
              >
                {ENCODER_LAYER_OPTIONS.map((layers) => (
                  <option key={layersToKey(layers)} value={layersToKey(layers)}>
                    [{layers.join(", ")}]
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Discriminator layers</span>
              <select
                value={analysisConfig.discriminatorLayersKey}
                onChange={(event) => onConfigChange({ discriminatorLayersKey: event.target.value })}
                disabled={analysisBusy}
              >
                {DISCRIMINATOR_LAYER_OPTIONS.map((layers) => (
                  <option key={layersToKey(layers)} value={layersToKey(layers)}>
                    [{layers.join(", ")}]
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Pretrain epoch</span>
              <input
                type="number"
                min="1"
                step="1"
                value={analysisConfig.pretrainEpoch}
                onChange={(event) => onConfigChange({ pretrainEpoch: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>

            <label>
              <span>Training epoch</span>
              <input
                type="number"
                min="1"
                step="1"
                value={analysisConfig.trainingEpoch}
                onChange={(event) => onConfigChange({ trainingEpoch: Number(event.target.value) })}
                disabled={analysisBusy}
              />
            </label>

            <label>
              <span>lambda_attention</span>
              <select
                value={analysisConfig.lambdaAttention}
                onChange={(event) => onConfigChange({ lambdaAttention: Number(event.target.value) })}
                disabled={analysisBusy}
              >
                {LAMBDA_ATTENTION_OPTIONS.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>

            <label>
              <span>n_clusters</span>
              <input
                type="text"
                value={analysisConfig.nClusters}
                placeholder="auto"
                onChange={(event) => onConfigChange({ nClusters: event.target.value })}
                disabled={analysisBusy}
              />
            </label>
          </div>

          <label className="checkbox-line">
            <input
              type="checkbox"
              checked={analysisConfig.isPca}
              onChange={(event) => onConfigChange({ isPca: event.target.checked })}
              disabled={analysisBusy}
            />
            <span>Use PCA before training (`is_pca=True` by default)</span>
          </label>

          {analysisConfig.lassoareMode === "reconstruct_embedding" ? (
            <div className="analysis-grid">
              <label>
                <span>Embedding</span>
                <select
                  value={analysisConfig.reconstructEmbeddingKey}
                  onChange={(event) => onConfigChange({ reconstructEmbeddingKey: event.target.value })}
                  disabled={analysisBusy || !availableEmbeddings.length}
                >
                  <option value="">Select an embedding</option>
                  {availableEmbeddings.map((key) => (
                    <option key={key} value={key}>{key}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>lambda_ref</span>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={analysisConfig.lambdaRef}
                  onChange={(event) => onConfigChange({ lambdaRef: Number(event.target.value) })}
                  disabled={analysisBusy}
                />
              </label>
            </div>
          ) : null}

          <p>
            `Generate` runs Lasso-ARE on `adata.X`. `Reconstruct Embedding` uses the selected embedding as input and
            automatically keeps `ref_enc_layers = enc_layers` and `ref_pretrain_epoch = pretrain_epoch`.
          </p>
          <div className="toolbar-group">
            <button
              type="button"
              onClick={onRunLassoare}
              disabled={analysisBusy || !selectedLassoareIds.length || (analysisConfig.lassoareMode === "reconstruct_embedding" && !analysisConfig.reconstructEmbeddingKey)}
            >
              Run Lasso-ARE
            </button>
            <button type="button" className="ghost-button" onClick={onShowLassoARECode} disabled={!selectedLassoareIds.length}>
              Show Code
            </button>
          </div>
        </div>

        <div className="analysis-status">
          <div className="analysis-header">
            <strong>Job status</strong>
            <span>{job ? job.status : "idle"}</span>
          </div>
          <p>{job?.message || "No analysis job has been started yet."}</p>
          {job ? <div className="progress-bar"><span style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} /></div> : null}
          {job?.status === "completed" ? (
            <div className="toolbar-group">
              <button type="button" className="ghost-button" onClick={() => onDownloadArtifact("result_h5ad")}>
                Download h5ad
              </button>
              {job?.result_info?.mapping_path ? (
                <button type="button" className="ghost-button" onClick={() => onDownloadArtifact("mapping")}>
                  Download mapping
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function App() {
  const [runtime, setRuntime] = useState({ profile: "cpu" });
  const [samples, setSamples] = useState([]);
  const [summary, setSummary] = useState(null);
  const [viewTwoSummary, setViewTwoSummary] = useState(null);
  const [viewTwoSource, setViewTwoSource] = useState({ analysisType: null, jobId: null, interactiveKind: null });
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Ready to explore a single-cell dataset.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedOnly, setSelectedOnly] = useState(true);
  const [openSection, setOpenSection] = useState("setting1");
  const [pendingIds, setPendingIds] = useState([]);
  const [viewTwoPendingIds, setViewTwoPendingIds] = useState([]);
  const [confirmedSelections, setConfirmedSelections] = useState([]);
  const [activeSelectionId, setActiveSelectionId] = useState(null);
  const [modalState, setModalState] = useState({ open: false, title: "", text: "" });
  const [copied, setCopied] = useState(false);
  const [analysisJob, setAnalysisJob] = useState(null);
  const [analysisConfig, setAnalysisConfig] = useState({
    sampleRate: 0.1,
    uniformRate: 0.5,
    leidenResolution: 1.0,
    doCorrect: true,
    lassoViewSelectionId: "",
    lassoareMode: "generate",
    lassoareSelectionIds: [],
    reconstructEmbeddingKey: "",
    encoderLayersKey: layersToKey([256, 64]),
    discriminatorLayersKey: layersToKey([256, 64]),
    pretrainEpoch: 20,
    trainingEpoch: 20,
    isPca: true,
    lambdaAttention: 0.1,
    lambdaRef: 0.3,
    nClusters: "",
  });
  const [markerConfig, setMarkerConfig] = useState({
    sourceView: "view1",
    obsCol: "",
    method: "t-test",
  });
  const [markerResult, setMarkerResult] = useState(null);
  const [viewTwoGeneExpression, setViewTwoGeneExpression] = useState("");

  const [viewOneConfig, setViewOneConfig] = useState(createViewConfig(null, null));
  const [viewTwoConfig, setViewTwoConfig] = useState(createViewConfig(null, null, { opacity: 0.92 }));
  const [viewOneData, setViewOneData] = useState(null);
  const [viewTwoData, setViewTwoData] = useState(null);

  const datasetId = summary?.dataset_id;
  const viewTwoDatasetId = viewTwoSummary?.dataset_id;
  const activeSelection = useMemo(
    () => confirmedSelections.find((selection) => selection.id === activeSelectionId) || confirmedSelections[confirmedSelections.length - 1] || null,
    [confirmedSelections, activeSelectionId],
  );
  const lassoViewSelection = useMemo(
    () => confirmedSelections.find((selection) => selection.id === analysisConfig.lassoViewSelectionId) || activeSelection,
    [confirmedSelections, analysisConfig.lassoViewSelectionId, activeSelection],
  );
  const lassoareSelections = useMemo(
    () => confirmedSelections.filter((selection) => (analysisConfig.lassoareSelectionIds || []).includes(selection.id)),
    [confirmedSelections, analysisConfig.lassoareSelectionIds],
  );
  const analysisRunning = Boolean(analysisJob && !["completed", "failed"].includes(analysisJob.status));
  const viewTwoInteractive = viewTwoSource.interactiveKind === "downsample";
  const noDefaultEmbedding = summary?.needs_umap_choice && !viewOneData;

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(apiUrl("/api/health")).then((response) => response.json()),
      fetch(apiUrl("/api/samples")).then((response) => response.json()),
    ])
      .then(([healthPayload, samplePayload]) => {
        if (!cancelled) {
          setRuntime(healthPayload);
          setSamples(samplePayload.samples || []);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSamples([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setAnalysisConfig((current) => {
      const validIds = new Set(confirmedSelections.map((selection) => selection.id));
      const nextLassoViewSelectionId = current.lassoViewSelectionId && validIds.has(current.lassoViewSelectionId)
        ? current.lassoViewSelectionId
        : (activeSelection?.id || confirmedSelections[0]?.id || "");
      let nextLassoareSelectionIds = (current.lassoareSelectionIds || []).filter((id) => validIds.has(id));
      if (!nextLassoareSelectionIds.length && confirmedSelections.length) {
        nextLassoareSelectionIds = confirmedSelections.map((selection) => selection.id);
      }

      const sameView = nextLassoViewSelectionId === current.lassoViewSelectionId;
      const sameLassoare = nextLassoareSelectionIds.length === (current.lassoareSelectionIds || []).length
        && nextLassoareSelectionIds.every((id, index) => id === current.lassoareSelectionIds[index]);
      if (sameView && sameLassoare) {
        return current;
      }
      return {
        ...current,
        lassoViewSelectionId: nextLassoViewSelectionId,
        lassoareSelectionIds: nextLassoareSelectionIds,
      };
    });
  }, [confirmedSelections, activeSelection?.id]);

  const duplicateStats = useMemo(() => {
    const occurrenceMap = new Map();
    confirmedSelections.forEach((selection) => {
      selection.ids.forEach((id) => {
        occurrenceMap.set(id, (occurrenceMap.get(id) || 0) + 1);
      });
    });

    let repeatedDistinct = 0;
    let repeatedAssignments = 0;
    occurrenceMap.forEach((count) => {
      if (count > 1) {
        repeatedDistinct += 1;
        repeatedAssignments += count - 1;
      }
    });

    const overlapCountMap = new Map();
    confirmedSelections.forEach((selection) => {
      let overlapCount = 0;
      selection.ids.forEach((id) => {
        if ((occurrenceMap.get(id) || 0) > 1) {
          overlapCount += 1;
        }
      });
      overlapCountMap.set(selection.id, overlapCount);
    });

    return {
      repeatedDistinct,
      repeatedAssignments,
      overlapCountMap,
    };
  }, [confirmedSelections]);

  const ingestResponse = (payload) => {
    const nextViewOneConfig = createViewConfig(payload.summary, payload.plot);
    const nextViewTwoConfig = createViewConfig(payload.summary, payload.plot, { opacity: 0.92 });

    setSummary(payload.summary);
    setViewTwoSummary(payload.summary);
    setViewTwoSource({ analysisType: null, jobId: null, interactiveKind: null });
    setPendingIds([]);
    setViewTwoPendingIds([]);
    setConfirmedSelections([]);
    setActiveSelectionId(null);
    setAnalysisJob(null);
    setAnalysisConfig((current) => ({
      ...current,
      reconstructEmbeddingKey: payload.summary?.default_embedding || payload.summary?.available_embeddings?.[0] || "",
    }));
    setMarkerConfig({
      sourceView: "view1",
      obsCol: payload.summary?.default_color_by || payload.summary?.obs_columns?.[0] || "",
      method: "t-test",
    });
    setMarkerResult(null);
    setViewTwoGeneExpression("");
    setViewOneConfig(nextViewOneConfig);
    setViewTwoConfig(nextViewTwoConfig);
    setViewOneData(payload.plot);
    setViewTwoData(payload.plot);
  };

  const fetchPlot = async (targetDatasetId, config) => {
    const response = await fetch(apiUrl(`/api/datasets/${targetDatasetId}/plot`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedding_key: config.embeddingKey || null,
        color_by: config.colorBy || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to update plot.");
    }
    return payload;
  };

  const fetchJob = async (jobId) => {
    const response = await fetch(apiUrl(`/api/analysis-jobs/${jobId}`));
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to fetch analysis job.");
    }
    return payload;
  };

  const fetchGeneExpressionPlot = async (targetDatasetId, gene, embeddingKey) => {
    const response = await fetch(apiUrl(`/api/datasets/${targetDatasetId}/gene-expression-plot`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gene,
        embedding_key: embeddingKey || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to generate gene expression plot.");
    }
    return payload;
  };

  const fetchMarkerPlot = async (targetDatasetId, obsCol, method) => {
    const response = await fetch(apiUrl(`/api/datasets/${targetDatasetId}/marker-plot`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        obs_col: obsCol,
        method: method || "t-test",
        top_n: 5,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to generate marker tables.");
    }
    return payload;
  };

  const applyCompletedJob = (jobSnapshot) => {
    if (!jobSnapshot.result_summary || !jobSnapshot.result_plot) {
      return;
    }
    const nextViewTwoConfig = createViewConfig(jobSnapshot.result_summary, jobSnapshot.result_plot, { opacity: 0.92 });
    setViewTwoSummary(jobSnapshot.result_summary);
    setViewTwoData(jobSnapshot.result_plot);
    setViewTwoConfig(nextViewTwoConfig);
    setViewTwoPendingIds([]);
    setViewTwoSource({
      analysisType: jobSnapshot.analysis_type,
      jobId: jobSnapshot.job_id,
      interactiveKind: jobSnapshot.interactive_kind || null,
    });
  };

  useEffect(() => {
    if (!analysisJob?.job_id || ["completed", "failed"].includes(analysisJob.status)) {
      return undefined;
    }

    let cancelled = false;
    const intervalId = window.setInterval(async () => {
      try {
        const snapshot = await fetchJob(analysisJob.job_id);
        if (cancelled) {
          return;
        }
        setAnalysisJob(snapshot);
        if (snapshot.status === "completed") {
          applyCompletedJob(snapshot);
          fetch(apiUrl("/api/health"))
            .then((response) => response.json())
            .then((healthPayload) => {
              if (!cancelled) {
                setRuntime(healthPayload);
              }
            })
            .catch(() => {});
          if (snapshot.analysis_type === "lasso_view" && Array.isArray(snapshot.result_info?.expanded_ids)) {
            commitConfirmedSelection(
              snapshot.result_info.expanded_ids,
              (selection) => `${selection.displayName} was added with ${selection.ids.length.toLocaleString()} propagated cells.`,
              {
                kind: "refining",
                displayLabelPrefix: "Refining",
                variablePrefix: "refine_list",
              },
            );
            return;
          }
          setStatus(`${snapshot.analysis_type} completed and loaded into View 2.`);
        } else if (snapshot.status === "failed") {
          setError(snapshot.error || snapshot.message || "Analysis failed.");
          setStatus("Analysis failed.");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      }
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [analysisJob?.job_id, analysisJob?.status]);

  const loadSample = async (sampleName) => {
    setBusy(true);
    setError("");
    setStatus(`Loading ${sampleName}...`);
    try {
      const response = await fetch(apiUrl(`/api/load-sample?name=${encodeURIComponent(sampleName)}`), { method: "POST" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load sample dataset.");
      }
      ingestResponse(payload);
      setSamples((current) => current.map((sample) => (
        sample.name === sampleName
          ? { ...sample, available: true, action: "load" }
          : sample
      )));
      setStatus(`Loaded ${payload.summary.dataset_name} with ${payload.summary.n_obs.toLocaleString()} cells.`);
    } catch (err) {
      setError(err.message);
      setStatus("Sample loading failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose an h5ad file first.");
      return;
    }

    setBusy(true);
    setError("");
    setStatus(`Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(apiUrl("/api/upload"), { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Upload failed.");
      }
      ingestResponse(payload);
      setStatus(`Loaded ${payload.summary.dataset_name} with ${payload.summary.n_obs.toLocaleString()} cells.`);
    } catch (err) {
      setError(err.message);
      setStatus("Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleViewConfigChange = async (viewName, patch) => {
    const currentConfig = viewName === "setting1" ? viewOneConfig : viewTwoConfig;
    const nextConfig = { ...currentConfig, ...patch };
    const setConfig = viewName === "setting1" ? setViewOneConfig : setViewTwoConfig;
    const setPlot = viewName === "setting1" ? setViewOneData : setViewTwoData;
    const targetDatasetId = viewName === "setting1" ? datasetId : viewTwoDatasetId;

    setConfig(nextConfig);

    if (!targetDatasetId) {
      return;
    }
    if (!("embeddingKey" in patch) && !("colorBy" in patch)) {
      return;
    }

    setBusy(true);
    setError("");
    try {
      const payload = await fetchPlot(targetDatasetId, nextConfig);
      setPlot(payload);
      setStatus(`Updated ${viewName} to ${payload.embedding_key}${payload.color_by ? ` colored by ${payload.color_by}` : ""}.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleComputeUmap = async () => {
    if (!datasetId) {
      return;
    }
    setBusy(true);
    setError("");
    setStatus("Computing UMAP with Scanpy...");
    try {
      const response = await fetch(apiUrl(`/api/datasets/${datasetId}/compute-umap`), { method: "POST" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "UMAP computation failed.");
      }
      ingestResponse(payload);
      setStatus("UMAP computed and displayed.");
    } catch (err) {
      setError(err.message);
      setStatus("UMAP computation failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleShowViewTwoGeneExpression = async () => {
    const gene = viewTwoGeneExpression.trim();
    if (!gene) {
      setError("Enter a gene name before showing expression in fig2.");
      return;
    }
    const targetDatasetId = viewTwoDatasetId || datasetId;
    if (!targetDatasetId) {
      setError("Load a dataset before showing gene expression.");
      return;
    }

    setBusy(true);
    setError("");
    setStatus(`Loading ${gene} expression in fig2...`);
    try {
      const payload = await fetchGeneExpressionPlot(targetDatasetId, gene, viewTwoConfig.embeddingKey || viewTwoSummary?.default_embedding || null);
      setViewTwoData(payload);
      setViewTwoConfig((current) => ({
        ...current,
        embeddingKey: payload.embedding_key || current.embeddingKey,
        colorBy: "",
      }));
      setViewTwoGeneExpression(payload.expression_gene || gene);
      setStatus(`Fig2 now shows ${payload.expression_gene || gene} expression intensity.`);
    } catch (err) {
      setError(err.message);
      setStatus("Gene expression plot failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleRunMarkerPlot = async (obsColOverride = "") => {
    const targetSummary = markerConfig.sourceView === "view2" ? (viewTwoSummary || summary) : summary;
    const targetDatasetId = markerConfig.sourceView === "view2" ? (viewTwoDatasetId || datasetId) : datasetId;
    const obsCol = obsColOverride || markerConfig.obsCol || targetSummary?.default_color_by || targetSummary?.obs_columns?.[0] || "";
    if (!targetDatasetId || !obsCol) {
      setError("Choose a figure and obs column before generating marker tables.");
      return;
    }

    setBusy(true);
    setError("");
    setStatus(`Generating marker tables for ${markerConfig.sourceView === "view2" ? "fig2" : "fig1"} by ${obsCol}...`);
    try {
      const payload = await fetchMarkerPlot(targetDatasetId, obsCol, markerConfig.method);
      setMarkerConfig((current) => ({ ...current, obsCol }));
      setMarkerResult(payload);
      setStatus(`Generated marker tables for ${payload.obs_col} with ${payload.genes.length.toLocaleString()} marker genes.`);
    } catch (err) {
      setError(err.message);
      setStatus("Marker table generation failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleAddSelectionIds = (ids) => {
    setPendingIds((current) => {
      const next = arrayUnion(current, ids);
      setStatus(`Draft selection now contains ${next.length.toLocaleString()} cells. Continue selecting, then Confirm or Cancel.`);
      return next;
    });
  };

  const handleAddViewTwoSelectionIds = (ids) => {
    setViewTwoPendingIds((current) => {
      const next = arrayUnion(current, ids);
      setStatus(`Downsample draft now contains ${next.length.toLocaleString()} sampled cells. Continue selecting or map them back to the original dataset.`);
      return next;
    });
  };

  const handleCancelPending = () => {
    setPendingIds([]);
    setStatus("Draft selection cleared.");
  };

  const commitConfirmedSelection = (ids, messageBuilder, options = {}) => {
    if (!ids.length) {
      return;
    }
    setConfirmedSelections((current) => {
      const kind = options.kind || "selection";
      const displayLabelPrefix = options.displayLabelPrefix || "Selection";
      const variablePrefix = options.variablePrefix || "select_list";
      const kindIndex = current.filter((selection) => selection.kind === kind).length;
      const displayName = options.displayName || `${displayLabelPrefix} ${kindIndex + 1}`;
      const nextSelection = {
        id: `selection-${Date.now()}-${current.length}`,
        kind,
        displayName,
        variableName: options.variableName || `${variablePrefix}${kindIndex + 1}`,
        ids: ids.slice(),
        color: SELECTION_COLORS[current.length % SELECTION_COLORS.length],
      };
      const next = [...current, nextSelection];
      setActiveSelectionId(nextSelection.id);
      setStatus(messageBuilder(nextSelection));
      return next;
    });
  };

  const handleConfirmPending = () => {
    if (!pendingIds.length) {
      return;
    }
    const nextIds = pendingIds.slice();
    commitConfirmedSelection(nextIds, (selection) => `Confirmed ${selection.variableName} with ${selection.ids.length.toLocaleString()} cells.`, {
      kind: "selection",
      displayLabelPrefix: "Selection",
      variablePrefix: "select_list",
    });
    setPendingIds([]);
  };

  const openTextModal = (title, text) => {
    setCopied(false);
    setModalState({ open: true, title, text });
  };

  const closeTextModal = () => {
    setModalState({ open: false, title: "", text: "" });
  };

  const copyModalContent = async () => {
    try {
      await navigator.clipboard.writeText(modalState.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      setError("Failed to copy selection content to the clipboard.");
    }
  };

  const formatSelectionLine = (selection, index) => `${selection.variableName || `select_list${index + 1}`}=[${selection.ids.join(",")}]`;

  const handleShowAllIds = () => {
    if (!confirmedSelections.length) {
      return;
    }
    openTextModal(
      "All selection classes",
      confirmedSelections.map((selection, index) => formatSelectionLine(selection, index)).join("\n"),
    );
  };

  const handleShowSingleSelection = (selection) => {
    const index = confirmedSelections.findIndex((item) => item.id === selection.id);
    openTextModal(`IDs for ${selection.variableName}`, formatSelectionLine(selection, index));
  };

  const handleDeleteSelection = (selectionId) => {
    setConfirmedSelections((current) => {
      const next = normalizeSelectionCatalog(current.filter((selection) => selection.id !== selectionId));
      setActiveSelectionId(next.length ? next[Math.max(0, next.length - 1)].id : null);
      return next;
    });
    setStatus("One confirmed selection class was deleted.");
  };

  const handleDeleteAllSelections = () => {
    setPendingIds([]);
    setViewTwoPendingIds([]);
    setConfirmedSelections([]);
    setActiveSelectionId(null);
    setStatus("All selection classes were removed.");
  };

  const exportSelectionJsonLines = () => {
    if (!confirmedSelections.length) {
      return;
    }
    const content = confirmedSelections.map((selection) => JSON.stringify(selection.ids)).join("\n");
    const blob = new Blob([content], { type: "application/x-ndjson" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "selection_lists.jsonl";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const submitAnalysisJob = async (payload, pendingMessage) => {
    if (!datasetId) {
      return;
    }
    setBusy(true);
    setError("");
    setStatus(pendingMessage);
    try {
      const response = await fetch(apiUrl(`/api/datasets/${datasetId}/analysis-jobs`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const jobSnapshot = await response.json();
      if (!response.ok) {
        throw new Error(jobSnapshot.detail || "Failed to create analysis job.");
      }
      setAnalysisJob(jobSnapshot);
      setViewTwoPendingIds([]);
      setViewTwoSource({
        analysisType: jobSnapshot.analysis_type,
        jobId: jobSnapshot.job_id,
        interactiveKind: null,
      });
      setStatus(jobSnapshot.message || "Analysis job queued.");
    } catch (err) {
      setError(err.message);
      setStatus("Analysis job failed to start.");
    } finally {
      setBusy(false);
    }
  };

  const handleRunLassoView = async () => {
    if (!lassoViewSelection) {
      setError("Choose one selection class for Lasso-View first.");
      return;
    }
    await submitAnalysisJob(
      {
        analysis_type: "lasso_view",
        selected_ids: lassoViewSelection.ids,
        embedding_key: viewOneConfig.embeddingKey || summary?.default_embedding || null,
        color_by: viewOneConfig.colorBy || summary?.default_color_by || null,
        obs_col: viewOneConfig.colorBy || summary?.default_color_by || null,
        leiden_resolution: analysisConfig.leidenResolution,
        do_correct: analysisConfig.doCorrect,
      },
      "Submitting Lasso-View job...",
    );
  };

  const handleRunDownsample = async () => {
    await submitAnalysisJob(
      {
        analysis_type: "downsample",
        embedding_key: viewOneConfig.embeddingKey || summary?.default_embedding || null,
        color_by: viewOneConfig.colorBy || summary?.default_color_by || null,
        sample_rate: analysisConfig.sampleRate,
        uniform_rate: analysisConfig.uniformRate,
        leiden_resolution: analysisConfig.leidenResolution,
      },
      "Submitting downsample job...",
    );
  };

  const handleRunLassoare = async () => {
    if (!lassoareSelections.length) {
      setError("Confirm at least one selection class before running LassoARE.");
      return;
    }
    if (analysisConfig.lassoareMode === "reconstruct_embedding" && !analysisConfig.reconstructEmbeddingKey) {
      setError("Choose an embedding before running reconstruction mode.");
      return;
    }
    await submitAnalysisJob(
      {
        analysis_type: "lassoare",
        lassoare_mode: analysisConfig.lassoareMode,
        selected_groups: lassoareSelections.map((selection) => selection.ids),
        embedding_key: analysisConfig.lassoareMode === "reconstruct_embedding" ? analysisConfig.reconstructEmbeddingKey : null,
        color_by: viewOneConfig.colorBy || summary?.default_color_by || null,
        leiden_resolution: analysisConfig.leidenResolution,
        n_clusters: analysisConfig.nClusters.trim() ? Number(analysisConfig.nClusters.trim()) : null,
        enc_layers: keyToLayers(analysisConfig.encoderLayersKey),
        disc_layers: keyToLayers(analysisConfig.discriminatorLayersKey),
        enc_pretrain_epoch: Number(analysisConfig.pretrainEpoch),
        disc_pretrain_epoch: Number(analysisConfig.pretrainEpoch),
        gan_epoch: Number(analysisConfig.trainingEpoch),
        is_pca: Boolean(analysisConfig.isPca),
        lambda_attention: Number(analysisConfig.lambdaAttention),
        lambda_ref: Number(analysisConfig.lambdaRef),
      },
      analysisConfig.lassoareMode === "generate" ? "Submitting Lasso-ARE generate job..." : "Submitting Lasso-ARE reconstruction job...",
    );
  };

  const handleShowLassoViewCode = () => openTextModal("Lasso-View code", buildLassoViewCode({ activeSelection: lassoViewSelection, summary, viewOneConfig, analysisConfig }));
  const handleShowDownsampleCode = () => openTextModal("Downsample code", buildDownsampleCode({ summary, viewOneConfig, analysisConfig }));
  const handleShowLassoARECode = () => openTextModal("Lasso-ARE code", buildLassoARECode({ confirmedSelections: lassoareSelections, analysisConfig }));

  const handleRecoverFromDownsample = async () => {
    if (!viewTwoSource.jobId || !viewTwoPendingIds.length) {
      return;
    }
    setBusy(true);
    setError("");
    setStatus("Recovering selected downsampled cells back to the original dataset...");
    try {
      const response = await fetch(apiUrl(`/api/analysis-jobs/${viewTwoSource.jobId}/recover-selection`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: viewTwoPendingIds }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to recover selection.");
      }
      const recoveredIds = payload.ids || [];
      commitConfirmedSelection(
        recoveredIds,
        (selection) => `Recovered ${selection.ids.length.toLocaleString()} original cells from the downsampled selection into ${selection.variableName}.`,
      );
      setViewTwoPendingIds([]);
    } catch (err) {
      setError(err.message);
      setStatus("Recovering downsampled selection failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleDownloadArtifact = (artifact) => {
    if (!analysisJob?.job_id) {
      return;
    }
    window.open(apiUrl(`/api/analysis-jobs/${analysisJob.job_id}/download/${artifact}`), "_blank");
  };

  const statusRows = [
    { label: "Dataset", value: summary?.dataset_name || "None loaded" },
    { label: "Cells", value: summary ? summary.n_obs.toLocaleString() : "0" },
    { label: "Genes", value: summary ? summary.n_vars.toLocaleString() : "0" },
    { label: "View 1", value: viewOneData ? `${viewOneData.embedding_key} | ${viewOneData.color_by || "No color"}` : "No plot" },
    { label: "View 2", value: viewTwoData ? `${viewTwoData.embedding_key} | ${viewTwoData.color_by || "No color"}` : "No plot" },
    { label: "View 2 dataset", value: viewTwoSummary?.dataset_name || "Original dataset" },
    { label: "Analysis job", value: analysisJob ? `${analysisJob.analysis_type} | ${analysisJob.status}` : "Idle" },
    { label: "Draft selection", value: `${pendingIds.length.toLocaleString()} cells pending` },
    { label: "Confirmed classes", value: `${confirmedSelections.length.toLocaleString()}` },
    {
      label: "Repeated selections",
      value: duplicateStats.repeatedDistinct
        ? `${duplicateStats.repeatedDistinct.toLocaleString()} repeated cells, ${duplicateStats.repeatedAssignments.toLocaleString()} extra assignments`
        : "No repeated cells across confirmed classes",
    },
  ];

  return (
    <main className="app-shell">
      <section className="workspace-shell">
        <section className="viewer-column">
          <div className="board-head panel-card">
            <div>
              <p className="eyebrow">Single-Cell and Spatial Transcriptomics Workspace</p>
              <h1>{summary?.dataset_name || "Lasso-ARE"}</h1>
              <p className="hero-copy">
                View 1 stays anchored to the original dataset. Analysis jobs create derived results that load into View 2, where downsampled outputs can be mapped back into new original-dataset selections.
              </p>
            </div>
            <div className="hero-badges">
              <span>Background jobs</span>
              <span>{runtimeLabel(runtime)}</span>
              <span>Selection classes</span>
            </div>
          </div>

          {noDefaultEmbedding ? (
            <section className="choice-banner panel-card">
              <div>
                <h3>No UMAP found</h3>
                <p>This dataset does not contain a usable UMAP embedding yet. You can compute UMAP with Scanpy or choose another two-dimensional entry in Setting 1.</p>
              </div>
              <button type="button" onClick={handleComputeUmap} disabled={busy || analysisRunning}>Compute UMAP</button>
            </section>
          ) : null}

          <section className="plot-grid">
            <PlotPanel
              title="View 1"
              subtitle="Interactive selection canvas on the original dataset"
              plotData={viewOneData}
              viewConfig={viewOneConfig}
              interactive={true}
              selectedOnly={false}
              pendingIds={pendingIds}
              confirmedSelections={confirmedSelections}
              onAddSelectionIds={handleAddSelectionIds}
              onCancelPending={handleCancelPending}
              onConfirmPending={handleConfirmPending}
              confirmButtonLabel="Confirm"
              draftNote="Each completed box or lasso action adds to the draft selection. Use Confirm to create a class, or Cancel to clear the draft."
            />
            <PlotPanel
              title="View 2"
              subtitle={viewTwoSource.analysisType ? `Derived result: ${viewTwoSource.analysisType}` : "Read-only comparison preview"}
              plotData={viewTwoData}
              viewConfig={viewTwoConfig}
              interactive={viewTwoInteractive}
              selectedOnly={viewTwoInteractive || viewTwoData?.color_mode === "continuous" ? false : selectedOnly}
              pendingIds={viewTwoPendingIds}
              confirmedSelections={viewTwoInteractive ? [] : confirmedSelections}
              onAddSelectionIds={viewTwoInteractive ? handleAddViewTwoSelectionIds : () => {}}
              onCancelPending={viewTwoInteractive ? () => setViewTwoPendingIds([]) : () => {}}
              onConfirmPending={viewTwoInteractive ? handleRecoverFromDownsample : () => {}}
              confirmButtonLabel={viewTwoInteractive ? "Map To Original Dataset" : "Confirm"}
              draftNote={viewTwoInteractive ? "Select sampled cells here, then map them back into a new confirmed selection class on the original dataset." : "Preview only."}
              statusHint={viewTwoSource.analysisType ? `Result from ${viewTwoSource.analysisType}` : "Preview only"}
            />
          </section>

          <SelectionTabs
            selections={confirmedSelections}
            activeSelectionId={activeSelectionId}
            overlapCountMap={duplicateStats.overlapCountMap}
            onSelect={setActiveSelectionId}
            onShowIds={handleShowSingleSelection}
            onDelete={handleDeleteSelection}
          />

          <AnalysisPanel
            summary={summary}
            viewTwoSummary={viewTwoSummary}
            activeSelection={activeSelection}
            confirmedSelections={confirmedSelections}
            analysisConfig={analysisConfig}
            onConfigChange={(patch) => setAnalysisConfig((current) => ({ ...current, ...patch }))}
            onRunLassoView={handleRunLassoView}
            onRunDownsample={handleRunDownsample}
            onRunLassoare={handleRunLassoare}
            markerConfig={markerConfig}
            onMarkerConfigChange={(patch) => setMarkerConfig((current) => ({ ...current, ...patch }))}
            onRunMarkerPlot={handleRunMarkerPlot}
            onShowLassoViewCode={handleShowLassoViewCode}
            onShowDownsampleCode={handleShowDownsampleCode}
            onShowLassoARECode={handleShowLassoARECode}
            onDownloadArtifact={handleDownloadArtifact}
            job={analysisJob}
            disabled={busy}
          />

          <MarkerBubblePlot markerResult={markerResult} />
        </section>

        <aside className="sidebar-column">
          <form className="side-card" onSubmit={handleUpload}>
            <div className="section-heading">
              <h2>Upload</h2>
              <p>Bring your own h5ad file or start from a local sample.</p>
            </div>
            <label className="file-drop">
              <input type="file" accept=".h5ad" onChange={(event) => setFile(event.target.files?.[0] || null)} />
              <span>{file ? file.name : "Choose a local .h5ad file"}</span>
            </label>
            <div className="button-stack">
              <button type="submit" disabled={busy || analysisRunning}>Upload h5ad file</button>
              {samples.map((sample) => (
                <button
                  key={sample.name}
                  type="button"
                  className="ghost-button"
                  onClick={() => loadSample(sample.name)}
                  disabled={busy || analysisRunning || sample.action === "unavailable"}
                >
                  {sample.label}: {sampleActionLabel(sample)}
                </button>
              ))}
            </div>
            <p className="status-line">{status}</p>
            {error ? <p className="error-line">{error}</p> : null}
          </form>

          <section className="side-card">
            <div className="section-heading">
              <h2>Current status</h2>
              <p>Each row wraps naturally so long metadata stays readable.</p>
            </div>
            <div className="status-list">
              {statusRows.map((row) => (
                <div key={row.label} className="status-row">
                  <span className="status-key">{row.label}</span>
                  <strong className="status-value">{row.value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className={classNames("side-card", "preview-mode-card", selectedOnly && "preview-mode-active")}>
            <div className="section-heading">
              <h2>Preview mode</h2>
              <p>This affects View 2 only when it is showing a non-interactive result.</p>
            </div>
            <button
              type="button"
              className={classNames("mode-button", selectedOnly && "mode-button-active")}
              onClick={() => setSelectedOnly((value) => !value)}
              disabled={viewTwoInteractive}
            >
              {selectedOnly ? "Selected only: ON" : "Selected only: OFF"}
            </button>
            <p className="mode-caption">
              When enabled, unselected cells turn grey in View 2. Confirmed classes still keep their dedicated colors.
            </p>
          </section>

          <section className="side-card">
            <div className="section-heading">
              <h2>Selection tools</h2>
              <p>Work with all confirmed selection classes at once.</p>
            </div>
            <div className="selection-summary">
              <strong>{confirmedSelections.length}</strong>
              <span>confirmed classes</span>
            </div>
            <div className="button-stack">
              <button type="button" className="ghost-button" onClick={handleShowAllIds} disabled={!confirmedSelections.length}>Show all IDs</button>
              <button type="button" className="ghost-button" onClick={exportSelectionJsonLines} disabled={!confirmedSelections.length}>Download JSON lines</button>
              <button type="button" className="ghost-button danger-button" onClick={handleDeleteAllSelections} disabled={!confirmedSelections.length && !pendingIds.length && !viewTwoPendingIds.length}>Delete all selections</button>
            </div>
          </section>

          <SettingsPanel
            title="Setting 1"
            open={openSection === "setting1"}
            onToggle={() => setOpenSection((value) => (value === "setting1" ? "" : "setting1"))}
            summary={summary}
            config={viewOneConfig}
            onConfigChange={(patch) => handleViewConfigChange("setting1", patch)}
            busy={busy || analysisRunning}
          />

          <SettingsPanel
            title="Setting 2"
            open={openSection === "setting2"}
            onToggle={() => setOpenSection((value) => (value === "setting2" ? "" : "setting2"))}
            summary={viewTwoSummary}
            config={viewTwoConfig}
            onConfigChange={(patch) => handleViewConfigChange("setting2", patch)}
            busy={busy}
            geneExpressionValue={viewTwoGeneExpression}
            onGeneExpressionChange={setViewTwoGeneExpression}
            onShowGeneExpression={handleShowViewTwoGeneExpression}
          />
        </aside>
      </section>

      <SelectionTextModal
        open={modalState.open}
        title={modalState.title}
        text={modalState.text}
        copied={copied}
        onClose={closeTextModal}
        onCopy={copyModalContent}
      />
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
