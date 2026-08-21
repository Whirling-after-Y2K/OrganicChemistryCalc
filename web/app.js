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

const viewerApp = createApp({
  data() {
    return {
      molecule: null,
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
    };
  },
  computed: {
    formulaHtml() {
      return this.molecule ? toSubscript(this.molecule.formula) : "";
    },
  },
  mounted() {
    this.resizeCanvas();
    this.initCanvasEvents();
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
        this.molecule = data.molecule;
        this.status = "已加载：" + (data.molecule.source || "");
        this.$nextTick(() => this.fitView());
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
        canvas.style.cursor = "grabbing";
      });
      window.addEventListener("mousemove", (event) => {
        if (!this.dragging) return;
        this.panX += event.clientX - this.lastX;
        this.panY += event.clientY - this.lastY;
        this.lastX = event.clientX;
        this.lastY = event.clientY;
        this.render();
      });
      window.addEventListener("mouseup", () => {
        this.dragging = false;
        canvas.style.cursor = "grab";
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
