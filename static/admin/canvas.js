const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_WIDTH = 164;
const NODE_HEIGHT = 96;
const H_SPACING = 260;
const V_SPACING = 160;
const MIN_SCALE = 0.35;
const MAX_SCALE = 2.8;

const prefixInput = document.getElementById("prefix-input");
const statusFilter = document.getElementById("status-filter");
const focusInput = document.getElementById("focus-input");
const focusButton = document.getElementById("focus-btn");
const viewCanvasButton = document.getElementById("view-canvas");
const viewListButton = document.getElementById("view-list");
const reloadButton = document.getElementById("reload-btn");
const resetRootButton = document.getElementById("reset-root");
const flashBanner = document.getElementById("flash");
const canvasPanel = document.getElementById("canvas-panel");
const listPanel = document.getElementById("list-panel");
const canvasContainer = canvasPanel.querySelector(".canvas-container");
const canvasToolbar = canvasContainer.querySelector(".canvas-toolbar");
const svg = document.getElementById("story-canvas");
const canvasGroup = document.getElementById("canvas-group");
const edgeLayer = document.getElementById("edge-layer");
const nodeLayer = document.getElementById("node-layer");
const zoomInButton = document.getElementById("zoom-in");
const zoomOutButton = document.getElementById("zoom-out");
const zoomResetButton = document.getElementById("zoom-reset");
const listTableBody = document.getElementById("list-table-body");
const inspector = document.getElementById("inspector");
const inspectorTitle = document.getElementById("inspector-title");
const inspectorStatus = document.getElementById("inspector-status");
const inspectorScenario = document.getElementById("inspector-scenario");
const inspectorUpdated = document.getElementById("inspector-updated");
const inspectorStorage = document.getElementById("inspector-storage");
const inspectorChildren = document.getElementById("inspector-children");
const inspectorVideo = document.getElementById("inspector-video");
const playVideoButton = document.getElementById("play-video");
const openModalButton = document.getElementById("open-modal");
const resetBranchButton = document.getElementById("reset-branch");
const previewOverlay = document.getElementById("preview-overlay");
const previewVideo = document.getElementById("preview-video");
const previewLinks = document.getElementById("preview-links");
const previewTitle = document.getElementById("preview-title");
const closePreviewButton = document.getElementById("close-preview");

const state = {
  nodes: new Map(),
  roots: [],
  collapsed: new Set(),
  detailCache: new Map(),
  selectedPath: null,
  baseDepth: 0,
  totalCount: 0,
  transform: { x: 0, y: 0, scale: 1 },
  transformDirty: false,
  drag: { active: false, id: null, startX: 0, startY: 0 },
  zoomRAF: null,
};

function setFlash(message, flavor = "info") {
  if (!message) {
    flashBanner.style.display = "none";
    flashBanner.textContent = "";
    return;
  }
  flashBanner.textContent = message;
  flashBanner.style.background = flavor === "error" ? "rgba(255,107,107,0.18)" : "rgba(61,139,253,0.18)";
  flashBanner.style.display = "block";
  window.setTimeout(() => {
    if (flashBanner.textContent === message) {
      flashBanner.style.display = "none";
      flashBanner.textContent = "";
    }
  }, 4200);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Accept": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail;
    try {
      detail = await response.json();
    } catch (error) {
      detail = {};
    }
    const message = detail?.detail || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json();
}

function sortPaths(a, b) {
  return a.path.localeCompare(b.path, undefined, { numeric: true });
}

function buildHierarchy(rawNodes, baseDepth) {
  state.nodes.clear();
  state.baseDepth = baseDepth;
  rawNodes.forEach((raw) => {
    const node = {
      path: raw.path,
      parentPath: raw.parent,
      depth: raw.depth,
      status: raw.status,
      scenario: raw.scenarioDisplay || "",
      posterUrl: raw.posterUrl || null,
      videoUrl: raw.videoUrl || null,
      contextVideoUrl: raw.contextVideoUrl || null,
      updatedAt: raw.updatedAt || null,
      childrenPaths: raw.children || [],
      children: [],
      element: null,
      imageEl: null,
      slot: null,
      posX: 0,
      posY: 0,
      hidden: false,
      collapsed: state.collapsed.has(raw.path),
    };
    state.nodes.set(node.path, node);
  });

  state.nodes.forEach((node) => {
    node.children = [];
  });

  state.nodes.forEach((node) => {
    const parent = node.parentPath ? state.nodes.get(node.parentPath) : null;
    node.parent = parent || null;
    if (parent) {
      parent.children.push(node);
    }
  });

  state.roots = [...state.nodes.values()].filter((node) => !node.parent);
  state.roots.sort(sortPaths);
}

function firstWalk(node, hiddenDueToAncestor) {
  const children = node.children.slice().sort(sortPaths);
  node.hidden = hiddenDueToAncestor;
  node.collapsed = state.collapsed.has(node.path);
  if (hiddenDueToAncestor) {
    children.forEach((child) => firstWalk(child, true));
    node.slot = null;
    return;
  }

  if (node.collapsed || children.length === 0) {
    children.forEach((child) => firstWalk(child, true));
    node.slot = firstWalk.nextSlot;
    firstWalk.nextSlot += 1;
    return;
  }

  children.forEach((child) => firstWalk(child, false));
  const firstChild = children[0];
  const lastChild = children[children.length - 1];
  node.slot = (firstChild.slot + lastChild.slot) / 2;
}

function computeLayout() {
  firstWalk.nextSlot = 0;
  state.roots.forEach((root) => firstWalk(root, false));

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  state.nodes.forEach((node) => {
    if (node.hidden || node.slot === null) {
      return;
    }
    const relativeDepth = Math.max(0, node.depth - state.baseDepth);
    node.centerX = relativeDepth * H_SPACING;
    node.centerY = node.slot * V_SPACING;
    node.drawX = node.centerX - NODE_WIDTH / 2;
    node.drawY = node.centerY - NODE_HEIGHT / 2;
    minX = Math.min(minX, node.centerX - NODE_WIDTH / 2);
    maxX = Math.max(maxX, node.centerX + NODE_WIDTH / 2);
    minY = Math.min(minY, node.centerY - NODE_HEIGHT / 2);
    maxY = Math.max(maxY, node.centerY + NODE_HEIGHT / 2);
  });

  if (!isFinite(minX)) {
    minX = 0;
    maxX = NODE_WIDTH;
    minY = 0;
    maxY = NODE_HEIGHT;
  }

  state.bounds = { minX, maxX, minY, maxY };
}

function createSvg(tag) {
  return document.createElementNS(SVG_NS, tag);
}

function clearLayer(layer) {
  while (layer.firstChild) {
    layer.removeChild(layer.firstChild);
  }
}

function renderEdges() {
  clearLayer(edgeLayer);
  state.nodes.forEach((node) => {
    if (node.hidden || !node.parent || node.parent.hidden || node.slot === null) {
      return;
    }
    const pathEl = createSvg("path");
    pathEl.classList.add("tree-edge");
    const startX = node.parent.centerX;
    const startY = node.parent.centerY;
    const endX = node.centerX;
    const endY = node.centerY;
    const midX = (startX + endX) / 2;
    const d = `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;
    pathEl.setAttribute("d", d);
    pathEl.dataset.parent = node.parent.path;
    pathEl.dataset.child = node.path;
    edgeLayer.appendChild(pathEl);
  });
}

function loadPoster(node) {
  if (!node.imageEl || node.posterLoaded || !node.posterUrl) {
    return;
  }
  node.imageEl.setAttributeNS(null, "href", node.posterUrl);
  node.imageEl.setAttributeNS("http://www.w3.org/1999/xlink", "href", node.posterUrl);
  node.posterLoaded = true;
}

function renderNodes() {
  clearLayer(nodeLayer);
  state.nodes.forEach((node) => {
    if (node.hidden || node.slot === null) {
      return;
    }
    const group = createSvg("g");
    group.classList.add("tree-node");
    group.dataset.path = node.path;
    group.dataset.status = node.status || "unknown";
    if (node.collapsed) {
      group.dataset.collapsed = "true";
    }
    if (state.selectedPath === node.path) {
      group.dataset.selected = "true";
    }
    group.setAttribute("transform", `translate(${node.drawX}, ${node.drawY})`);

    const backdrop = createSvg("rect");
    backdrop.classList.add("backdrop");
    backdrop.setAttribute("width", NODE_WIDTH);
    backdrop.setAttribute("height", NODE_HEIGHT);
    backdrop.setAttribute("rx", 16);
    backdrop.setAttribute("ry", 16);
    group.appendChild(backdrop);

    const image = createSvg("image");
    image.setAttribute("width", NODE_WIDTH);
    image.setAttribute("height", NODE_HEIGHT);
    image.setAttribute("preserveAspectRatio", "xMidYMid slice");
    if (node.posterUrl) {
      image.dataset.src = node.posterUrl;
    }
    group.appendChild(image);
    node.imageEl = image;
    node.posterLoaded = false;

    const overlay = createSvg("rect");
    overlay.setAttribute("width", NODE_WIDTH);
    overlay.setAttribute("height", NODE_HEIGHT);
    overlay.setAttribute("rx", 16);
    overlay.setAttribute("ry", 16);
    overlay.setAttribute("fill", "url(#none)");
    overlay.style.fill = "rgba(8, 12, 20, 0.32)";
    group.appendChild(overlay);

    const titleBg = createSvg("rect");
    titleBg.setAttribute("x", 10);
    titleBg.setAttribute("y", 12);
    titleBg.setAttribute("width", NODE_WIDTH - 20);
    titleBg.setAttribute("height", 24);
    titleBg.setAttribute("rx", 12);
    titleBg.setAttribute("fill", "rgba(5, 8, 15, 0.68)");
    group.appendChild(titleBg);

    const title = createSvg("text");
    title.setAttribute("x", 20);
    title.setAttribute("y", 30);
    title.setAttribute("fill", "#f6f8fa");
    title.setAttribute("font-size", "13");
    title.setAttribute("font-family", "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif");
    title.textContent = node.path || "<root>";
    group.appendChild(title);

    const statusBadge = createSvg("text");
    statusBadge.setAttribute("x", 20);
    statusBadge.setAttribute("y", NODE_HEIGHT - 16);
    statusBadge.setAttribute("fill", "rgba(246,248,250,0.82)");
    statusBadge.setAttribute("font-size", "12");
    statusBadge.setAttribute("font-family", "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif");
    statusBadge.textContent = node.status || "unknown";
    group.appendChild(statusBadge);

    group.addEventListener("click", (event) => {
      event.stopPropagation();
      selectNode(node.path, { center: true });
    });

    group.addEventListener("dblclick", (event) => {
      event.stopPropagation();
      toggleCollapse(node.path);
    });

    node.element = group;
    nodeLayer.appendChild(group);
  });
}

function updateVisibleThumbnails() {
  const rect = canvasContainer.getBoundingClientRect();
  const padding = 200;
  state.nodes.forEach((node) => {
    if (!node.imageEl || node.posterLoaded || !node.posterUrl || node.hidden || node.slot === null) {
      return;
    }
    const screenX = state.transform.x + node.centerX * state.transform.scale;
    const screenY = state.transform.y + node.centerY * state.transform.scale;
    const visible =
      screenX > -padding &&
      screenY > -padding &&
      screenX < rect.width + padding &&
      screenY < rect.height + padding;
    if (visible) {
      loadPoster(node);
    }
  });
}

function applyTransform() {
  canvasGroup.setAttribute(
    "transform",
    `translate(${state.transform.x}, ${state.transform.y}) scale(${state.transform.scale})`
  );
  updateVisibleThumbnails();
  state.transformDirty = false;
}

function requestTransform() {
  if (state.transformDirty) {
    return;
  }
  state.transformDirty = true;
  state.zoomRAF = requestAnimationFrame(applyTransform);
}

function centerOnBounds() {
  const rect = canvasContainer.getBoundingClientRect();
  const { minX, maxX, minY, maxY } = state.bounds || { minX: 0, maxX: NODE_WIDTH, minY: 0, maxY: NODE_HEIGHT };
  const width = Math.max(maxX - minX, NODE_WIDTH);
  const height = Math.max(maxY - minY, NODE_HEIGHT);
  const scaleX = rect.width / (width + 220);
  const scaleY = rect.height / (height + 220);
  const newScale = Math.min(Math.max(Math.min(scaleX, scaleY), MIN_SCALE), 1.4);
  state.transform.scale = newScale;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  state.transform.x = rect.width / 2 - centerX * newScale;
  state.transform.y = rect.height / 2 - centerY * newScale;
  requestTransform();
}

function centerOnNode(path) {
  const node = state.nodes.get(path);
  if (!node || node.hidden || node.slot === null) {
    setFlash("Node is hidden or missing in current view", "error");
    return;
  }
  const rect = canvasContainer.getBoundingClientRect();
  state.transform.x = rect.width / 2 - node.centerX * state.transform.scale;
  state.transform.y = rect.height / 2 - node.centerY * state.transform.scale;
  requestTransform();
}

function renderTree() {
  computeLayout();
  renderEdges();
  renderNodes();
  updateVisibleThumbnails();
}

async function loadTree({ showMessage = true } = {}) {
  try {
    const params = new URLSearchParams();
    const prefix = prefixInput.value.trim();
    if (prefix) params.set("prefix", prefix);
    const status = statusFilter.value;
    if (status) params.set("status", status);
    params.set("limit", "200");
    const data = await fetchJSON(`/admin/api/tree?${params.toString()}`);
    state.totalCount = data.total;
    buildHierarchy(data.nodes || [], data.baseDepth || 0);
    state.detailCache.clear();
    if (!state.nodes.size) {
      nodeLayer.innerHTML = "";
      edgeLayer.innerHTML = "";
      setFlash("No scenes found. Try adjusting your filters.");
      updateListView();
      inspectorTitle.textContent = "Select a node";
      inspectorStatus.style.display = "none";
      inspectorScenario.textContent = "Hover or select a branch to preview prompts and media.";
      inspectorUpdated.textContent = "";
      inspectorStorage.textContent = "";
      inspectorChildren.innerHTML = "";
      inspectorVideo.removeAttribute("src");
      inspectorVideo.load();
      return;
    }
    renderTree();
    centerOnBounds();
    updateListView();
    if (showMessage) {
      const shown = Math.min(state.nodes.size, data.total);
      setFlash(`Loaded ${shown} scene${shown === 1 ? "" : "s"} (total ${data.total}).`);
    }
  } catch (error) {
    setFlash(error.message, "error");
  }
}

function clearSelection() {
  if (state.selectedPath) {
    const previous = state.nodes.get(state.selectedPath);
    if (previous && previous.element) {
      delete previous.element.dataset.selected;
    }
  }
  state.selectedPath = null;
}

async function selectNode(path, { center = false } = {}) {
  const node = state.nodes.get(path);
  if (!node || node.hidden || node.slot === null) {
    setFlash("Node not visible in current tree", "error");
    return;
  }
  clearSelection();
  state.selectedPath = node.path;
  if (node.element) {
    node.element.dataset.selected = "true";
  }
  if (center) {
    centerOnNode(node.path);
  }
  updateInspector(node);
}

function updateInspector(node) {
  inspectorTitle.textContent = node.path || "Root scene";
  if (node.status) {
    inspectorStatus.textContent = node.status;
    inspectorStatus.dataset.status = node.status;
    inspectorStatus.style.display = "inline-flex";
  } else {
    inspectorStatus.style.display = "none";
  }
  inspectorScenario.textContent = node.scenario || "No scenario recorded.";
  inspectorUpdated.textContent = node.updatedAt ? `Updated: ${new Date(node.updatedAt).toLocaleString()}` : "";
  inspectorStorage.textContent = "Loading storage…";
  inspectorChildren.innerHTML = "";
  if (node.videoUrl) {
    inspectorVideo.style.display = "block";
    inspectorVideo.poster = node.posterUrl || "";
  } else {
    inspectorVideo.style.display = "none";
    inspectorVideo.removeAttribute("src");
    inspectorVideo.load();
  }

  const cached = state.detailCache.get(node.path);
  if (cached) {
    fillInspectorDetails(node, cached);
    return;
  }
  fetchJSON(`/admin/api/scene?path=${encodeURIComponent(node.path)}`)
    .then((detail) => {
      state.detailCache.set(node.path, detail);
      if (state.selectedPath === node.path) {
        fillInspectorDetails(node, detail);
      }
    })
    .catch((error) => {
      inspectorStorage.textContent = error.message;
    });
}

function fillInspectorDetails(node, detail) {
  const bytes = detail?.storageBytes ?? 0;
  const mb = bytes ? (bytes / (1024 * 1024)).toFixed(2) : "0.00";
  inspectorStorage.textContent = `Storage footprint: ${mb} MiB`;
  inspectorChildren.innerHTML = "";
  const list = detail?.children || [];
  if (!list.length) {
    const div = document.createElement("div");
    div.className = "child-row";
    div.textContent = "No children recorded.";
    inspectorChildren.appendChild(div);
  } else {
    list.forEach((child) => {
      const row = document.createElement("div");
      row.className = "child-row";
      const left = document.createElement("div");
      left.textContent = child.path;
      const right = document.createElement("span");
      right.className = "status-pill";
      right.dataset.status = child.status;
      right.textContent = child.status;
      row.append(left, right);
      row.addEventListener("click", () => {
        selectNode(child.path, { center: true });
      });
      inspectorChildren.appendChild(row);
    });
  }
}

function toggleCollapse(path) {
  if (state.collapsed.has(path)) {
    state.collapsed.delete(path);
  } else {
    state.collapsed.add(path);
  }
  renderTree();
  requestTransform();
}

async function handleReset(path) {
  const label = path ? `branch ${path}` : "entire world";
  let confirmMessage = `Delete ${label}? This removes all clips, metrics, and scene records for this branch.`;
  try {
    const detail = await fetchJSON(`/admin/api/scene?path=${encodeURIComponent(path || "")}`);
    const bytes = (detail.storageBytes || 0) / (1024 * 1024);
    confirmMessage += `\n\nStorage ≈ ${bytes.toFixed(2)} MiB`;
  } catch (error) {
    // ignore detail errors
  }
  if (!window.confirm(confirmMessage)) {
    return;
  }
  try {
    await fetchJSON("/admin/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path || "", inclusive: true }),
    });
    pruneSubtree(path);
    renderTree();
    requestTransform();
    updateListView();
    setFlash(`Deleted ${label}.`);
    if (!state.nodes.has(state.selectedPath || "")) {
      clearSelection();
      inspectorTitle.textContent = "Select a node";
      inspectorScenario.textContent = "Hover or select a branch to preview prompts and media.";
      inspectorUpdated.textContent = "";
      inspectorStorage.textContent = "";
      inspectorChildren.innerHTML = "";
      inspectorVideo.removeAttribute("src");
      inspectorVideo.load();
    }
  } catch (error) {
    setFlash(error.message, "error");
  }
}

function pruneSubtree(path) {
  const target = path || "";
  const toRemove = [];
  const toRemoveSet = new Set();
  state.nodes.forEach((node, key) => {
    if (key === target || key.startsWith(`${target}/`)) {
      toRemove.push(key);
      toRemoveSet.add(key);
    }
  });
  toRemove.forEach((key) => {
    state.nodes.delete(key);
    state.collapsed.delete(key);
    state.detailCache.delete(key);
  });
  state.nodes.forEach((node) => {
    if (!node.childrenPaths) return;
    node.childrenPaths = node.childrenPaths.filter((childPath) => !toRemoveSet.has(childPath));
  });
  buildHierarchy(Array.from(state.nodes.values()).map((node) => ({
    path: node.path,
    parent: node.parentPath,
    depth: node.depth,
    status: node.status,
    scenarioDisplay: node.scenario,
    posterUrl: node.posterUrl,
    videoUrl: node.videoUrl,
    contextVideoUrl: node.contextVideoUrl,
    updatedAt: node.updatedAt,
    children: node.childrenPaths,
  })), state.baseDepth);
}

function updateListView() {
  listTableBody.innerHTML = "";
  if (!state.nodes.size) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5" style="padding:24px 16px; text-align:center; color: var(--muted);">No data loaded.</td>';
    listTableBody.appendChild(row);
    return;
  }
  const nodes = Array.from(state.nodes.values()).sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
  nodes.forEach((node) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${node.path || "<root>"}</td>
      <td><span class="status-pill" data-status="${node.status || "unknown"}">${node.status || "unknown"}</span></td>
      <td>${node.updatedAt ? new Date(node.updatedAt).toLocaleString() : "—"}</td>
      <td>${node.scenario ? node.scenario.slice(0, 120) + (node.scenario.length > 120 ? "…" : "") : "+"}</td>
      <td style="display:flex; gap:6px; flex-wrap:wrap;">
        <button type="button" class="chip" data-action="select">Select</button>
        <button type="button" class="chip" data-action="center">Center</button>
        <button type="button" class="chip" data-action="preview" ${node.videoUrl ? "" : "disabled"}>Preview</button>
        <button type="button" class="danger" data-action="reset">Reset</button>
      </td>
    `;
    tr.dataset.path = node.path;
    listTableBody.appendChild(tr);
  });
}

function switchView(view) {
  if (view === "canvas") {
    canvasPanel.classList.remove("hidden");
    listPanel.classList.add("hidden");
    viewCanvasButton.dataset.active = "true";
    viewListButton.dataset.active = "false";
  } else {
    canvasPanel.classList.add("hidden");
    listPanel.classList.remove("hidden");
    viewCanvasButton.dataset.active = "false";
    viewListButton.dataset.active = "true";
  }
}

function handlePointerDown(event) {
  if (event.button !== 0) return;
  const target = event.target.closest && event.target.closest(".tree-node");
  if (target) {
    return;
  }
  state.drag.active = true;
  state.drag.id = event.pointerId;
  state.drag.startX = event.clientX;
  state.drag.startY = event.clientY;
  svg.setPointerCapture(event.pointerId);
}

function handlePointerMove(event) {
  if (!state.drag.active || event.pointerId !== state.drag.id) {
    return;
  }
  const dx = event.clientX - state.drag.startX;
  const dy = event.clientY - state.drag.startY;
  state.drag.startX = event.clientX;
  state.drag.startY = event.clientY;
  state.transform.x += dx;
  state.transform.y += dy;
  requestTransform();
}

function handlePointerUp(event) {
  if (!state.drag.active || event.pointerId !== state.drag.id) {
    return;
  }
  state.drag.active = false;
  state.drag.id = null;
  svg.releasePointerCapture(event.pointerId);
}

function handleWheel(event) {
  if (event.ctrlKey) {
    return;
  }
  event.preventDefault();
  const rect = canvasContainer.getBoundingClientRect();
  const offsetX = event.clientX - rect.left;
  const offsetY = event.clientY - rect.top;
  const scaleFactor = event.deltaY < 0 ? 1.12 : 0.88;
  const newScale = Math.min(Math.max(state.transform.scale * scaleFactor, MIN_SCALE), MAX_SCALE);
  const canvasX = (offsetX - state.transform.x) / state.transform.scale;
  const canvasY = (offsetY - state.transform.y) / state.transform.scale;
  state.transform.scale = newScale;
  state.transform.x = offsetX - canvasX * newScale;
  state.transform.y = offsetY - canvasY * newScale;
  requestTransform();
}

function zoom(delta) {
  const rect = canvasContainer.getBoundingClientRect();
  const offsetX = rect.width / 2;
  const offsetY = rect.height / 2;
  const scaleFactor = delta > 0 ? 1.2 : 0.83;
  const newScale = Math.min(Math.max(state.transform.scale * scaleFactor, MIN_SCALE), MAX_SCALE);
  const canvasX = (offsetX - state.transform.x) / state.transform.scale;
  const canvasY = (offsetY - state.transform.y) / state.transform.scale;
  state.transform.scale = newScale;
  state.transform.x = offsetX - canvasX * newScale;
  state.transform.y = offsetY - canvasY * newScale;
  requestTransform();
}

function resetZoom() {
  centerOnBounds();
}

function playInspectorVideo() {
  if (!state.selectedPath) return;
  const node = state.nodes.get(state.selectedPath);
  if (!node || !node.videoUrl) {
    setFlash("No video for this node", "error");
    return;
  }
  inspectorVideo.crossOrigin = "anonymous";
  inspectorVideo.src = node.videoUrl;
  inspectorVideo.poster = node.posterUrl || "";
  inspectorVideo.muted = false;
  const playPromise = inspectorVideo.play();
  if (playPromise && typeof playPromise.then === "function") {
    playPromise.catch(() => {
      inspectorVideo.load();
      inspectorVideo.play().catch(() => {/* swallow */});
    });
  }
}

function openModalForSelected() {
  if (!state.selectedPath) return;
  const node = state.nodes.get(state.selectedPath);
  if (!node || !node.videoUrl) {
    setFlash("No video to preview", "error");
    return;
  }
  previewVideo.crossOrigin = "anonymous";
  previewVideo.src = node.videoUrl;
  previewVideo.poster = node.posterUrl || "";
  previewTitle.textContent = node.path || "Root";
  previewLinks.innerHTML = `
    <a href="${node.videoUrl}" target="_blank" rel="noopener" style="color: var(--accent);">Open video in new tab</a>
    ${node.contextVideoUrl ? ' · <a href="' + node.contextVideoUrl + '" target="_blank" rel="noopener" style="color: var(--accent);">Full continuity</a>' : ''}
  `;
  previewOverlay.classList.add("active");
  const playPromise = previewVideo.play();
  if (playPromise && typeof playPromise.then === "function") {
    playPromise.catch(() => {
      previewVideo.load();
      previewVideo.play().catch(() => {/* swallow */});
    });
  }
  document.addEventListener("keydown", escListener);
}

function closeModal() {
  previewOverlay.classList.remove("active");
  previewVideo.pause();
  previewVideo.removeAttribute("src");
  previewVideo.load();
  previewLinks.textContent = "";
  document.removeEventListener("keydown", escListener);
}

function escListener(event) {
  if (event.key === "Escape") {
    closeModal();
  }
}

function initEvents() {
  svg.addEventListener("pointerdown", handlePointerDown);
  svg.addEventListener("pointermove", handlePointerMove);
  svg.addEventListener("pointerup", handlePointerUp, true);
  svg.addEventListener("pointerleave", handlePointerUp, true);
  svg.addEventListener("wheel", handleWheel, { passive: false });
  svg.addEventListener("dblclick", (event) => event.preventDefault());

  zoomInButton.addEventListener("click", () => zoom(1));
  zoomOutButton.addEventListener("click", () => zoom(-1));
  zoomResetButton.addEventListener("click", resetZoom);

  viewCanvasButton.addEventListener("click", () => switchView("canvas"));
  viewListButton.addEventListener("click", () => switchView("list"));

  reloadButton.addEventListener("click", () => loadTree({ showMessage: false }));
  prefixInput.addEventListener("change", () => loadTree());
  statusFilter.addEventListener("change", () => loadTree());

  canvasContainer.addEventListener("click", (event) => {
    if (event.target.closest && event.target.closest(".tree-node")) {
      return;
    }
    clearSelection();
    inspectorTitle.textContent = "Select a node";
    inspectorStatus.style.display = "none";
    inspectorScenario.textContent = "Hover or select a branch to preview prompts and media.";
    inspectorUpdated.textContent = "";
    inspectorStorage.textContent = "";
    inspectorChildren.innerHTML = "";
    inspectorVideo.removeAttribute("src");
    inspectorVideo.load();
  });

  canvasPanel.addEventListener("click", (event) => event.stopPropagation());
  inspector.addEventListener("click", (event) => event.stopPropagation());
  if (canvasToolbar) {
    canvasToolbar.addEventListener("click", (event) => event.stopPropagation());
  }

  focusButton.addEventListener("click", () => {
    const path = focusInput.value.trim();
    if (!path && state.nodes.has("")) {
      selectNode("", { center: true });
      return;
    }
    if (!state.nodes.has(path)) {
      setFlash("Path not loaded in current tree", "error");
      return;
    }
    selectNode(path, { center: true });
  });

  playVideoButton.addEventListener("click", playInspectorVideo);
  openModalButton.addEventListener("click", openModalForSelected);
  resetBranchButton.addEventListener("click", () => handleReset(state.selectedPath || ""));
  resetRootButton.addEventListener("click", () => handleReset(""));

  previewOverlay.addEventListener("click", (event) => {
    if (event.target === previewOverlay) {
      closeModal();
    }
  });
  closePreviewButton.addEventListener("click", closeModal);

  listTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const row = button.closest("tr");
    if (!row) return;
    const path = row.dataset.path;
    switch (button.dataset.action) {
      case "select":
        selectNode(path, { center: false });
        break;
      case "center":
        selectNode(path, { center: true });
        switchView("canvas");
        break;
      case "preview":
        selectNode(path, { center: false });
        openModalForSelected();
        break;
      case "reset":
        handleReset(path);
        break;
      default:
        break;
    }
  });

  document.addEventListener("keydown", (event) => {
    const tag = event.target.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea") {
      return;
    }
    if (event.key === "+" || event.key === "=") {
      zoom(1);
    } else if (event.key === "-" || event.key === "_") {
      zoom(-1);
    } else if (event.key === "0") {
      resetZoom();
    } else if (event.key === "ArrowLeft") {
      state.transform.x += 80;
      requestTransform();
    } else if (event.key === "ArrowRight") {
      state.transform.x -= 80;
      requestTransform();
    } else if (event.key === "ArrowUp") {
      state.transform.y += 80;
      requestTransform();
    } else if (event.key === "ArrowDown") {
      state.transform.y -= 80;
      requestTransform();
    }
  });
}

initEvents();
loadTree();
