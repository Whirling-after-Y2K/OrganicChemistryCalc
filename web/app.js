/* 有机分子查看器前端：Vue 3 + Canvas 渲染。 */
"use strict";

const { createApp } = Vue;

const ELEMENT_COLORS = {
  c: "#333333", n: "#1f5fb0", o: "#d22",
  h: "#777777", f: "#2a9d2a", cl: "#0f7a0f",
  br: "#8b3a3a", i: "#6b2f9e",
};
const ACTIVE_H_COLOR = "#ff8c00";
const BOND_COLOR = "#222222";
// 布局键长（与 oc_render.BOND_LEN 保持一致）
const BOND_LEN = 40;
// 原子字号 = 键长 / 1.5，保证任意缩放下 键长 : 原子大小 ≈ 1.5 : 1
const ATOM_FONT_RATIO = BOND_LEN / 2;

function toSubscript(text) {
  const digits = "₀₁₂₃₄₅₆₇₈₉";
  return text.replace(/(\d+)/g, (m) => [...m].map((d) => digits[+d]).join(""));
}

function hashString(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

const ELEMENT_OPTIONS = [
  { value: "c", label: "C" },
  { value: "n", label: "N" },
  { value: "o", label: "O" },
  { value: "f", label: "F" },
  { value: "cl", label: "Cl" },
  { value: "br", label: "Br" },
  { value: "i", label: "I" },
  { value: "h", label: "H" },
];

const viewerApp = createApp({
  data() {
    return {
      tabs: [],
      activeTabId: null,
      nextTabId: 1,
      editMode: "select",
      elementOptions: ELEMENT_OPTIONS,
      elementChoice: "c",
      bondOrder: 1,
      pendingAtom: null,
      showSaveDialog: false,
      savePath: "",
      error: "",
      status: "就绪",
      pathText: "",
      zoom: 1,
      panX: 0,
      panY: 0,
      cssW: 0,
      cssH: 0,
      dragging: false,
      lastX: 0,
      lastY: 0,
      dragStartX: 0,
      dragStartY: 0,
      dragMoved: false,
    };
  },
  computed: {
    formulaHtml() {
      return this.molecule ? toSubscript(this.molecule.formula) : "";
    },
    molecule() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      return tab ? tab.molecule : null;
    },
    activeSource() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      return tab ? tab.source : "";
    },
  },
  mounted() {
    this.resizeCanvas();
    this.initCanvasEvents();
    this.updateCursor();
    window.addEventListener("resize", () => this.resizeCanvas());
    const params = new URLSearchParams(location.search);
    const openPath = params.get("open");
    if (openPath) {
      this.pathText = openPath;
      this.loadByPath();
    }
  },
  methods: {
    // ---- 文件加载 ----
    pickFile() {
      this.$refs.fileInput.click();
    },
    onDropFile(event) {
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) this.loadFile(file);
    },
    onFileChosen(event) {
      const file = event.target.files && event.target.files[0];
      event.target.value = "";
      if (file) this.loadFile(file);
    },
    async loadFile(file) {
      this.error = "";
      this.status = "正在读取 " + file.name + " …";
      try {
        const content = await file.text();
        await this.requestLoad({ filename: file.name, content });
      } catch (err) {
        this.error = "读取文件失败：" + err.message;
        this.status = "";
      }
    },
    async loadByPath() {
      const path = (this.pathText || "").trim();
      if (!path) {
        this.error = "请输入分子文件路径";
        return;
      }
      await this.requestLoad({ path });
    },
    async requestLoad(payload) {
      this.error = "";
      this.status = "正在加载…";
      try {
        const response = await fetch("/api/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!data.ok) {
          this.error = data.error || "加载失败";
          this.status = "";
          return;
        }
        const loaded = data.molecule;
        const source = loaded.source || "";
        // 去重键：路径加载用完整路径（忽略大小写），文件上传用 文件名+内容哈希，
        // 避免不同目录下同名文件被误判为同一分子
        let key = source;
        let displaySource = source;
        if (typeof payload.path === "string") {
          const normPath = payload.path.replace(/\\/g, "/");
          key = normPath.toLowerCase();
          const parts = normPath.split("/");
          displaySource =
            parts.length >= 2
              ? parts[parts.length - 2] + "\\" + parts[parts.length - 1]
              : source;
        } else if (
          typeof payload.filename === "string" &&
          typeof payload.content === "string"
        ) {
          key = payload.filename.toLowerCase() + ":" + hashString(payload.content);
        }
        const existing = this.tabs.find((t) => t.key === key);
        if (existing) {
          this.activeTabId = existing.id;
          this.status = "已激活：" + source;
          this.$nextTick(() => this.fitView());
        } else {
          const tab = {
            id: this.nextTabId++,
            key: key,
            source: displaySource,
            name: loaded.name || "",
            formula: loaded.formula || "",
            molecule: loaded,
            sessionId: data.session_id || "",
            originalPath: typeof payload.path === "string" ? payload.path : "",
            dirty: false,
          };
          this.tabs.push(tab);
          this.activeTabId = tab.id;
          this.status = "已加载：" + source;
          this.$nextTick(() => this.fitView());
        }
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
      }
    },

    // ---- 标签页 ----
    switchTab(id) {
      if (id === this.activeTabId) return;
      this.activeTabId = id;
      this.$nextTick(() => this.fitView());
    },
    closeTab(id) {
      const index = this.tabs.findIndex((t) => t.id === id);
      if (index < 0) return;
      const wasActive = id === this.activeTabId;
      this.tabs.splice(index, 1);
      if (wasActive) {
        const next = this.tabs[Math.min(index, this.tabs.length - 1)];
        this.activeTabId = next ? next.id : null;
      }
      this.render();
      this.$nextTick(() => this.fitView());
    },

    // ---- 编辑模式 ----
    setEditMode(mode) {
      this.editMode = mode;
      this.pendingAtom = null;
      this.updateCursor();
      this.render();
    },
    updateCursor() {
      const canvas = this.$refs.canvas;
      if (!canvas) return;
      if (this.editMode === "select") {
        canvas.style.cursor = this.dragging ? "grabbing" : "grab";
      } else if (this.editMode === "delete") {
        canvas.style.cursor = "pointer";
      } else {
        canvas.style.cursor = "crosshair";
      }
    },
    async editApi(opData) {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab || !tab.sessionId) {
        this.error = "没有可编辑的分子";
        this.status = "";
        return false;
      }
      this.error = "";
      this.status = "正在编辑…";
      try {
        const response = await fetch("/api/edit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: tab.sessionId, ...opData }),
        });
        const data = await response.json();
        if (!data.ok) {
          this.error = data.error || "编辑失败";
          this.status = "";
          return false;
        }
        tab.molecule = data.molecule;
        tab.dirty = true;
        this.$nextTick(() => this.fitView());
        return true;
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
        return false;
      }
    },
    atomHitRadius(atom) {
      const fontSize = Math.max(4, ATOM_FONT_RATIO * this.zoom);
      // 判定区域与原子本身同大（直径 ≈ 字号）
      return Math.max(12, fontSize * 0.5);
    },
    hitAtom(x, y) {
      let best = null;
      let bestDistance = Infinity;
      for (const atom of this.molecule.atoms) {
        const radius = this.atomHitRadius(atom);
        const sx = this.panX + atom.x * this.zoom;
        const sy = this.panY + atom.y * this.zoom;
        const d = Math.hypot(sx - x, sy - y);
        if (d <= radius && d <= bestDistance) {
          bestDistance = d;
          best = atom.id;
        }
      }
      return best;
    },
    hitBond(x, y) {
      const threshold = 8;
      for (const bond of this.molecule.bonds) {
        const a = this.molecule.atoms[bond.a];
        const c = this.molecule.atoms[bond.b];
        const x1 = this.panX + a.x * this.zoom;
        const y1 = this.panY + a.y * this.zoom;
        const x2 = this.panX + c.x * this.zoom;
        const y2 = this.panY + c.y * this.zoom;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const len2 = dx * dx + dy * dy;
        let t = len2 ? ((x - x1) * dx + (y - y1) * dy) / len2 : 0;
        t = Math.max(0, Math.min(1, t));
        const px = x1 + t * dx - x;
        const py = y1 + t * dy - y;
        if (Math.hypot(px, py) <= threshold) return bond;
      }
      return null;
    },
    handleCanvasClick(event) {
      if (!this.molecule || !this.molecule.atoms.length) return;
      if (this.editMode === "select") return;
      const canvas = this.$refs.canvas;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const atomId = this.hitAtom(x, y);
      if (this.editMode === "addAtom") {
        if (atomId !== null) {
          this.pendingAtom = atomId;
          this.status = "已选锚点，点击空白处添加原子";
          this.render();
        } else if (this.pendingAtom !== null) {
          this.addAtomAtAnchor();
        } else {
          this.status = "请先点击一个原子作为锚点";
        }
        return;
      }
      if (this.editMode === "addBond") {
        if (atomId !== null) {
          if (this.pendingAtom === null) {
            this.pendingAtom = atomId;
            this.status = "已选第一个原子，再点第二个原子";
            this.render();
          } else if (this.pendingAtom === atomId) {
            this.status = "请选择另一个原子";
          } else {
            this.addBondBetween(this.pendingAtom, atomId);
          }
        } else {
          this.pendingAtom = null;
          this.render();
        }
        return;
      }
      if (this.editMode === "delete") {
        if (atomId !== null) {
          this.deleteAtom(atomId);
        } else {
          const bond = this.hitBond(x, y);
          if (bond) this.deleteBond(bond);
          else this.status = "点击原子或键线进行删除";
        }
      }
    },
    async addAtomAtAnchor() {
      const ok = await this.editApi({
        op: "add_atom_bonded",
        atom: this.pendingAtom,
        element: this.elementChoice,
        order: 1,
      });
      this.pendingAtom = null;
      if (ok) this.status = "已添加原子并成键";
    },
    async addBondBetween(atom1, atom2) {
      const molecule = this.molecule;
      const existing = molecule.bonds.find(
        (b) =>
          (b.a === atom1 && b.b === atom2) || (b.a === atom2 && b.b === atom1)
      );
      if (existing && existing.order === this.bondOrder) {
        this.pendingAtom = null;
        this.status = "该键已是此键级";
        return;
      }
      const ok = await this.editApi({
        op: "set_bond_order",
        atom1: atom1,
        atom2: atom2,
        order: this.bondOrder,
      });
      this.pendingAtom = null;
      if (ok) this.status = "已设置键级";
    },
    async deleteAtom(atomId) {
      const ok = await this.editApi({ op: "del_atom", atom: atomId });
      this.pendingAtom = null;
      if (ok) this.status = "已删除原子";
    },
    async deleteBond(bond) {
      const ok = await this.editApi({
        op: "del_bond",
        atom1: bond.a,
        atom2: bond.b,
      });
      this.pendingAtom = null;
      if (ok) this.status = "已删除键";
    },
    openSaveDialog() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab) return;
      this.savePath =
        tab.originalPath ||
        "demo_output\\" + (tab.formula || "molecule") + ".py";
      this.showSaveDialog = true;
    },
    async confirmSave() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab) return;
      const path = this.savePath.trim();
      if (!path) {
        this.error = "请输入保存路径";
        return;
      }
      this.error = "";
      this.status = "正在保存…";
      try {
        const response = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: tab.sessionId, path: path }),
        });
        const data = await response.json();
        if (!data.ok) {
          this.error = data.error || "保存失败";
          this.status = "";
          return;
        }
        tab.dirty = false;
        this.status = "已保存到：" + data.path;
        this.showSaveDialog = false;
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
      }
    },

    // ---- 画布尺寸与事件 ----
    resizeCanvas() {
      const canvas = this.$refs.canvas;
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.cssW = rect.width;
      this.cssH = rect.height;
      canvas.width = Math.max(1, rect.width * dpr);
      canvas.height = Math.max(1, rect.height * dpr);
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.render();
    },
    initCanvasEvents() {
      const canvas = this.$refs.canvas;
      canvas.addEventListener(
        "wheel",
        (event) => {
          event.preventDefault();
          const factor = event.deltaY < 0 ? 1.12 : 0.9;
          const rect = canvas.getBoundingClientRect();
          const mx = event.clientX - rect.left;
          const my = event.clientY - rect.top;
          this.panX = mx - (mx - this.panX) * factor;
          this.panY = my - (my - this.panY) * factor;
          this.zoom = Math.min(8, Math.max(0.08, this.zoom * factor));
          this.render();
        },
        { passive: false }
      );
      canvas.addEventListener("mousedown", (event) => {
        this.dragging = true;
        this.lastX = event.clientX;
        this.lastY = event.clientY;
        this.dragStartX = event.clientX;
        this.dragStartY = event.clientY;
        this.dragMoved = false;
        this.updateCursor();
      });
      window.addEventListener("mousemove", (event) => {
        if (!this.dragging) return;
        if (
          !this.dragMoved &&
          Math.hypot(
            event.clientX - this.dragStartX,
            event.clientY - this.dragStartY
          ) > 5
        ) {
          this.dragMoved = true;
        }
        if (this.dragMoved) {
          this.panX += event.clientX - this.lastX;
          this.panY += event.clientY - this.lastY;
          this.render();
        }
        this.lastX = event.clientX;
        this.lastY = event.clientY;
      });
      window.addEventListener("mouseup", (event) => {
        if (this.dragging && !this.dragMoved) {
          this.handleCanvasClick(event);
        }
        this.dragging = false;
        this.updateCursor();
      });
    },

    // ---- 视图 ----
    zoomBy(dir) {
      const factor = dir > 0 ? 1.25 : 0.8;
      const cx = this.cssW / 2;
      const cy = this.cssH / 2;
      this.panX = cx - (cx - this.panX) * factor;
      this.panY = cy - (cy - this.panY) * factor;
      this.zoom = Math.min(8, Math.max(0.08, this.zoom * factor));
      this.render();
    },
    fitView() {
      if (!this.molecule || !this.molecule.atoms.length) return;
      const xs = this.molecule.atoms.map((a) => a.x);
      const ys = this.molecule.atoms.map((a) => a.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const bw = Math.max(maxX - minX, 1);
      const bh = Math.max(maxY - minY, 1);
      const margin = 100;
      this.zoom = Math.min(
        (this.cssW - 2 * margin) / bw,
        (this.cssH - 2 * margin) / bh,
        4
      );
      this.panX = this.cssW / 2 - ((minX + maxX) / 2) * this.zoom;
      this.panY = this.cssH / 2 - ((minY + maxY) / 2) * this.zoom;
      this.render();
    },

    // ---- 渲染 ----
    render() {
      const canvas = this.$refs.canvas;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, this.cssW, this.cssH);
      if (!this.molecule || !this.molecule.atoms.length) {
        this.drawEmpty(ctx);
        return;
      }
      const transform = (x, y) => [this.panX + x * this.zoom, this.panY + y * this.zoom];
      for (const bond of this.molecule.bonds) {
        const p1 = this.molecule.atoms[bond.a];
        const p2 = this.molecule.atoms[bond.b];
        this.drawBond(ctx, p1, p2, bond.display_order || bond.order || 1, transform);
      }
      for (const atom of this.molecule.atoms) {
        this.drawAtom(ctx, atom, transform);
      }
      if (this.pendingAtom !== null && this.molecule.atoms[this.pendingAtom]) {
        const anchor = this.molecule.atoms[this.pendingAtom];
        const [px, py] = transform(anchor.x, anchor.y);
        ctx.strokeStyle = "#e74c3c";
        ctx.lineWidth = Math.max(1.5, 2 * this.zoom);
        ctx.beginPath();
        ctx.arc(px, py, this.atomHitRadius(anchor), 0, Math.PI * 2);
        ctx.stroke();
      }
    },
    drawEmpty(ctx) {
      ctx.fillStyle = "#999";
      ctx.font = "14px 'Microsoft YaHei', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("请打开一个分子文件", this.cssW / 2, this.cssH / 2);
    },
    drawBond(ctx, p1, p2, order, transform) {
      const [x1, y1] = transform(p1.x, p1.y);
      const [x2, y2] = transform(p2.x, p2.y);
      const z = this.zoom;
      const dx = x2 - x1, dy = y2 - y1;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const px = -uy, py = ux;
      const gap = 3.5 * z;
      const offsets = order === 1 ? [0] : order === 2 ? [-1, 1] : [-1, 0, 1];
      ctx.strokeStyle = BOND_COLOR;
      ctx.lineWidth = Math.max(0.5, 2 * z);
      ctx.lineCap = "round";
      for (const off of offsets) {
        ctx.beginPath();
        ctx.moveTo(x1 + px * off * gap, y1 + py * off * gap);
        ctx.lineTo(x2 + px * off * gap, y2 + py * off * gap);
        ctx.stroke();
      }
    },
    drawAtom(ctx, atom, transform) {
      const [x, y] = transform(atom.x, atom.y);
      const color = atom.active ? ACTIVE_H_COLOR : ELEMENT_COLORS[atom.element] || "#333";
      const label = atom.label || atom.element.toUpperCase();
      const fontSize = Math.max(4, ATOM_FONT_RATIO * this.zoom);
      ctx.font = "600 " + fontSize + "px 'Segoe UI', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      // 不透明背景椭圆：完全遮住穿过后方的键线（含字母间空隙与边缘）
      const textWidth = ctx.measureText(label).width;
      const pad = fontSize * 0.08;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.ellipse(x, y, textWidth / 2 + pad, fontSize / 2 + pad, 0, 0, Math.PI * 2);
      ctx.fill();
      // 白色描边兜底：防止字形抗锯齿边缘透出键线
      ctx.lineWidth = Math.max(0.3, fontSize * 0.03);
      ctx.strokeStyle = "#ffffff";
      ctx.strokeText(label, x, y);
      ctx.fillStyle = color;
      ctx.fillText(label, x, y);
    },
  },
});
window.__viewer = viewerApp.mount("#app");
