/* 有机分子查看器前端：Vue 3 + Canvas 渲染。
   编辑操作统一走 /api/edit：add_atom / add_atom_bonded / add_benzene / add_nitro /
   add_bond / set_bond_order / del_atom / del_bond。
   苯环、硝基需先在"选择"模式点选锚点原子；删除 π 体系成员原子时后端会移除整个
   π 体系，但只删除被点击的那个原子，其余成员保留。
   加键模式：新建键走 add_bond，π 体系成员与体系外原子可以直接成键；已有键的
   键级调整走 set_bond_order，涉及 π 体系时后端会拒绝。 */
"use strict";

const { createApp } = Vue;

const CANVAS_THEME = {
  light: {
    elementColors: {
      c: "#333333", n: "#1f5fb0", o: "#d22",
      h: "#777777", f: "#2a9d2a", cl: "#0f7a0f",
      br: "#8b3a3a", i: "#6b2f9e",
    },
    activeH: "#ff8c00",
    bond: "#222222",
    mask: "#ffffff",
    empty: "#999999",
    pending: "#e74c3c",
  },
  dark: {
    elementColors: {
      c: "#e7eaee", n: "#8ab4f8", o: "#ff7b72",
      h: "#aab4bd", f: "#7fd18c", cl: "#85e0a3",
      br: "#ffb59a", i: "#d0a6ff",
    },
    activeH: "#ffb84d",
    bond: "#e0e4e8",
    mask: "#17191C",
    empty: "#8d99a6",
    pending: "#ff5252",
  },
};
// 布局键长（与 oc_render.BOND_LEN 保持一致）
const BOND_LEN = 40;
// 原子字号 = 键长 / 1.5，保证任意缩放下 键长 : 原子大小 ≈ 1.5 : 1
const ATOM_FONT_RATIO = BOND_LEN / 2;
// 分子文件后缀：内容仍是 Python 构建脚本，需与 oc_io.MOLECULE_SUFFIX 一致
const MOLECULE_SUFFIX = ".mol";

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

const TEMPLATE_OPTIONS = {
  benzene: { displaySource: "苯" },
  nitro: { displaySource: "硝基甲烷" },
};

const viewerApp = createApp({
  data() {
    return {
      tabs: [],
      activeTabId: null,
      nextTabId: 1,
      analysisTab: "info",
      editMode: "select",
      elementOptions: ELEMENT_OPTIONS,
      elementChoice: "c",
      bondOrder: 1,
      pendingAtom: null,
      showSaveDialog: false,
      savePath: "",
      error: "",
      status: "就绪",
      loadRequests: {},
      analysisJobs: [],
      dialogStartedAt: 0,
      fileDialogMs: null,
      fileReadMs: null,
      groupOptions: [],
      isomerGroups: [],
      isomerHydrogenPattern: "",
      isomerAllowExtraRings: false,
      isomerLimit: 200,
      isomerResults: [],
      isomerTotal: 0,
      isomerTruncated: false,
      isomerLoading: false,
      synthesisReactantIds: [],
      synthesisTargetId: null,
      synthesisReaction: "",
      synthesisConditions: "",
      synthesisMaxSteps: 4,
      synthesisMaxRoutes: 5,
      synthesisDedupeStrategy: true,
      synthesisOptimalOnly: true,
      synthesisRoutes: [],
      synthesisLoading: false,
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
      theme: "light",
    };
  },
  computed: {
    activeTab() {
      return this.tabs.find((t) => t.id === this.activeTabId) || null;
    },
    formulaHtml() {
      return this.molecule ? toSubscript(this.molecule.formula) : "";
    },
    equivalentHydrogenText() {
      const groups = this.molecule?.equivalent_hydrogen_groups;
      return groups?.length ? groups.join(" : ") : "无";
    },
    molecule() {
      return this.activeTab ? this.activeTab.molecule : null;
    },
    activeSource() {
      return this.activeTab ? this.activeTab.source : "";
    },
    synthesisReady() {
      return Boolean(this.synthesisReactantIds.length && this.synthesisTargetId);
    },
  },
  mounted() {
    this.initializeTheme();
    this.resizeCanvas();
    this.initCanvasEvents();
    this.updateCursor();
    window.addEventListener("resize", () => this.resizeCanvas());
    // 刷新或关闭标签页时取消仍在跑的后台分析任务：纯 CPU 密集的搜索会
    // 持续与后续请求争抢 GIL，让“打开分子”之类的操作看起来卡住。
    window.addEventListener("pagehide", () => this.cancelAnalysisJobs());
    this.loadGroupOptions();
    const params = new URLSearchParams(location.search);
    const openPath = params.get("open");
    if (openPath) {
      this.requestLoad({ path: openPath });
    }
  },
  methods: {
    initializeTheme() {
      this.theme = "dark";
      document.documentElement.dataset.theme = this.theme;
      this.render();
    },
    toggleTheme() {
      this.theme = this.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = this.theme;
      this.render();
    },
    canvasTheme() {
      return CANVAS_THEME[this.theme] || CANVAS_THEME.light;
    },
    async readJson(response) {
      const contentType = response.headers.get("Content-Type") || "";
      if (!contentType.toLowerCase().includes("application/json")) {
        throw new Error(`请求失败（HTTP ${response.status}）`);
      }
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
      }
      return data;
    },
    async getJson(url) {
      const response = await fetch(url);
      return this.readJson(response);
    },
    async postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      return this.readJson(response);
    },
    async loadGroupOptions() {
      try {
        const response = await fetch("/api/groups");
        const data = await response.json();
        this.groupOptions = data.ok ? data.groups : [];
      } catch (err) {
        this.groupOptions = [];
      }
    },
    // ---- 文件加载 ----
    pickFile() {
      // 记录点击时刻：系统文件对话框这一段不经过任何接口，却是用户
      // 感知“卡住”的主要区间（首次打开、杀毒软件扫描文件时尤其慢）。
      this.dialogStartedAt = performance.now();
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
      // 从点击“打开分子”到对话框返回的耗时；此前完全没有任何计时覆盖。
      if (this.dialogStartedAt) {
        this.fileDialogMs = Math.round(performance.now() - this.dialogStartedAt);
        this.dialogStartedAt = 0;
      } else {
        this.fileDialogMs = null;
      }
      try {
        const readStart = performance.now();
        const content = await file.text();
        this.fileReadMs = Math.round(performance.now() - readStart);
        await this.requestLoad({ filename: file.name, content });
      } catch (err) {
        this.error = "读取文件失败：" + err.message;
        this.status = "";
      }
    },
    // ---- 导入到当前标签页 ----
    pickImportFile() {
      this.$refs.importInput.click();
    },
    onImportFileChosen(event) {
      const file = event.target.files && event.target.files[0];
      event.target.value = "";
      if (file) this.importFile(file);
    },
    async importFile(file) {
      this.error = "";
      this.status = "正在读取 " + file.name + " …";
      let content = "";
      try {
        content = await file.text();
      } catch (err) {
        this.error = "读取文件失败：" + err.message;
        this.status = "";
        return;
      }
      const tab = this.activeTab;
      if (!tab || !tab.sessionId) {
        // 没有打开的分子时，导入等同于打开：新建一个标签页
        await this.requestLoad({ filename: file.name, content });
        return;
      }
      this.status = "正在导入…";
      try {
        const response = await fetch("/api/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: tab.sessionId,
            content,
          }),
        });
        const data = await response.json();
        if (!data.ok) {
          this.error = data.error || "导入失败";
          this.status = "";
          return;
        }
        tab.molecule = data.molecule;
        tab.formula = data.molecule.formula || tab.formula;
        tab.dirty = true;
        // 原子集合变了，旧的锚点选择失效
        this.pendingAtom = null;
        this.status = "已导入 " + file.name + " 到当前标签页";
        this.render();
        this.$nextTick(() => this.fitView());
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
      }
    },
    async openNewMolecule() {
      this.error = "";
      const key = "new:" + performance.now() + ":" + Math.random().toString(36).slice(2);
      await this.performLoad({ path: "new.mol" }, {
        key,
        displaySource: "new.mol",
      });
    },
    async duplicateMolecule() {
      const tab = this.activeTab;
      if (!tab || !tab.sessionId) {
        this.error = "没有可复制的分子";
        this.status = "";
        return;
      }
      this.error = "";
      this.status = "正在复制…";
      try {
        const response = await fetch("/api/duplicate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: tab.sessionId }),
        });
        const data = await response.json();
        if (!data.ok) {
          this.error = data.error || "复制失败";
          this.status = "";
          return;
        }
        const loaded = data.molecule;
        const copy = {
          id: this.nextTabId++,
          // 每次复制都拿到新 session_id，因此 key 唯一，不会并入已有标签页
          key: "copy:" + data.session_id,
          source: (tab.source || loaded.source || "") + " 副本",
          name: loaded.name || "",
          formula: loaded.formula || "",
          molecule: loaded,
          sessionId: data.session_id,
          originalPath: "",
          dirty: false,
        };
        this.tabs.push(copy);
        this.activeTabId = copy.id;
        this.status = "已复制：" + copy.source;
        this.$nextTick(() => this.fitView());
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
      }
    },
    async openTemplate(templateName) {
      const option = TEMPLATE_OPTIONS[templateName];
      if (!option) {
        this.error = "未知分子模板";
        return;
      }
      this.error = "";
      const key =
        "template:" + templateName + ":" +
        performance.now() + ":" + Math.random().toString(36).slice(2);
      await this.performLoad({}, {
        key,
        displaySource: option.displaySource,
      }, "/api/templates/" + templateName);
    },
    loadDescriptor(payload) {
      if (typeof payload.path === "string") {
        const normPath = payload.path.replace(/\\/g, "/");
        const parts = normPath.split("/");
        const source = parts[parts.length - 1] || payload.path;
        return {
          key: normPath.toLowerCase(),
          displaySource:
            parts.length >= 2
              ? parts[parts.length - 2] + "\\" + parts[parts.length - 1]
              : source,
        };
      }
      const source =
        typeof payload.filename === "string" ? payload.filename : "分子";
      const content = typeof payload.content === "string" ? payload.content : "";
      return {
        key: source.toLowerCase() + ":" + hashString(content),
        displaySource: source,
      };
    },
    async requestLoad(payload) {
      this.error = "";
      const descriptor = this.loadDescriptor(payload);
      const existing = this.tabs.find((t) => t.key === descriptor.key);
      if (existing) {
        this.activeTabId = existing.id;
        this.status = "已激活：" + existing.source;
        this.$nextTick(() => this.fitView());
        return;
      }

      const pending = this.loadRequests[descriptor.key];
      if (pending) return pending;

      const request = this.performLoad(payload, descriptor);
      this.loadRequests[descriptor.key] = request;
      try {
        return await request;
      } finally {
        if (this.loadRequests[descriptor.key] === request) {
          delete this.loadRequests[descriptor.key];
        }
      }
    },
    async performLoad(payload, descriptor, url = "/api/load") {
      this.status = "正在加载…";
      const startedAt = performance.now();
      try {
        const response = await fetch(url, {
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
        const tab = {
          id: this.nextTabId++,
          key: descriptor.key,
          source: descriptor.displaySource || source,
          name: loaded.name || "",
          formula: loaded.formula || "",
          molecule: loaded,
          sessionId: data.session_id || "",
          originalPath: typeof payload.path === "string" ? payload.path : "",
          dirty: false,
        };
        this.tabs.push(tab);
        this.activeTabId = tab.id;
        if (!this.synthesisTargetId) this.synthesisTargetId = tab.sessionId;
        if (!this.synthesisReactantIds.length) {
          this.synthesisReactantIds = [tab.sessionId];
        }
        const elapsedMs = Math.max(0, Math.round(performance.now() - startedAt));
        this.status = `已加载：${source}（${elapsedMs} ms）`;
        // file_dialog_ms / file_read_ms 覆盖 fetch 之前的盲区：系统对话框
        // 首次打开、杀毒软件扫描文件时，这一段比后端耗时更容易造成卡顿感。
        const fileTiming = {};
        if (this.fileDialogMs !== null) fileTiming.file_dialog_ms = this.fileDialogMs;
        if (this.fileReadMs !== null) fileTiming.file_read_ms = this.fileReadMs;
        this.fileDialogMs = null;
        this.fileReadMs = null;
        console.info("分子加载耗时", {
          source,
          frontend_ms: elapsedMs,
          ...fileTiming,
          ...(data.timing || {}),
        });
        this.$nextTick(() => this.fitView());
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
      }
    },

    // ---- 分析结果与流程 ----
    openAnalysisMolecule(item) {
      const key = "analysis:" + item.session_id;
      const existing = this.tabs.find((t) => t.key === key);
      if (existing) {
        this.activeTabId = existing.id;
        this.$nextTick(() => this.fitView());
        return;
      }
      const molecule = item.molecule;
      const source = molecule.source || item.formula || "分析结果";
      const tab = {
        id: this.nextTabId++,
        key: key,
        source: source,
        name: molecule.name || "",
        formula: molecule.formula || item.formula || "",
        molecule: molecule,
        sessionId: item.session_id,
        originalPath: "",
        dirty: false,
      };
      this.tabs.push(tab);
      this.activeTabId = tab.id;
      this.$nextTick(() => this.fitView());
    },
    parseHydrogenPattern() {
      const text = this.isomerHydrogenPattern.trim();
      if (!text) return [];
      const values = text
        .split(/[，,、\s]+/)
        .filter(Boolean)
        .map((token) => Number(token));
      if (values.some((value) => !Number.isInteger(value) || value <= 0)) {
        throw new Error("等位氢模式需为正整数，如 1,1,2,3");
      }
      return values;
    },
    wait(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    },
    cancelAnalysisJobs() {
      // 取消仍在跑的后台分析任务；sendBeacon 在页面卸载时仍能可靠发出请求
      for (const jobId of this.analysisJobs.slice()) {
        const url = `/api/analysis-jobs/${jobId}/cancel`;
        if (navigator.sendBeacon) navigator.sendBeacon(url);
        else fetch(url, { method: "POST" }).catch(() => {});
      }
    },
    async pollAnalysisJob(jobId, intervalMs, updateProgress) {
      // 登记在跑的任务：页面卸载时据此通知后端取消，避免后台线程继续
      // 占用 CPU，与后续“打开分子”之类的请求争抢 GIL。
      if (!this.analysisJobs.includes(jobId)) this.analysisJobs.push(jobId);
      try {
        for (;;) {
          const data = await this.getJson(`/api/analysis-jobs/${jobId}`);
          if (data.status === "done") return data.result;
          if (data.status === "failed") throw new Error(data.error || "分析失败");
          if (data.status === "cancelled") throw new Error("分析已取消");
          if (updateProgress) updateProgress(data);
          await this.wait(intervalMs || 250);
        }
      } finally {
        this.analysisJobs = this.analysisJobs.filter((id) => id !== jobId);
      }
    },
    async analyzeIsomers() {
      const tab = this.activeTab;
      if (!tab || !tab.sessionId) return;
      this.error = "";
      this.status = "正在枚举同分异构体…";
      this.isomerLoading = true;
      try {
        const equivalent_hydrogens = this.parseHydrogenPattern();
        const submitted = await this.postJson("/api/isomers/jobs", {
          session_id: tab.sessionId,
          required_groups: this.isomerGroups,
          equivalent_hydrogens,
          allow_extra_rings: this.isomerAllowExtraRings ? true : null,
          limit: this.isomerLimit,
        });
        const data = await this.pollAnalysisJob(
          submitted.job_id,
          submitted.poll_interval_ms,
          (job) => {
            const seconds = (job.elapsed_ms / 1000).toFixed(1);
            this.status = `正在枚举同分异构体…（${seconds}s）`;
          },
        );
        this.isomerResults = data.isomers;
        this.isomerTotal = data.total;
        this.isomerTruncated = data.truncated;
        this.status =
          "同分异构体：" +
          data.total +
          (data.truncated ? `（显示前 ${data.returned} 个）` : "");
      } catch (err) {
        this.error = err.message;
        this.status = "";
        this.isomerResults = [];
        this.isomerTotal = 0;
        this.isomerTruncated = false;
      } finally {
        this.isomerLoading = false;
      }
    },
    toggleSynthesisReactant(sessionId, event) {
      const selected = new Set(this.synthesisReactantIds);
      if (event.target.checked) selected.add(sessionId);
      else selected.delete(sessionId);
      this.synthesisReactantIds = [...selected];
    },
    async planSynthesis() {
      if (!this.synthesisReady) {
        this.error = "请选择起始反应物和目标产物";
        return;
      }
      this.error = "";
      this.status = "正在规划合成路线…";
      this.synthesisLoading = true;
      try {
        const submitted = await this.postJson("/api/synthesis/jobs", {
          reactant_ids: this.synthesisReactantIds,
          target_id: this.synthesisTargetId,
          reaction: this.synthesisReaction,
          conditions: this.synthesisConditions,
          max_steps: this.synthesisMaxSteps,
          max_routes: this.synthesisMaxRoutes,
          dedupe_strategy: this.synthesisDedupeStrategy,
          optimal_only: this.synthesisOptimalOnly,
        });
        const data = await this.pollAnalysisJob(
          submitted.job_id,
          submitted.poll_interval_ms,
          (job) => {
            const seconds = (job.elapsed_ms / 1000).toFixed(1);
            this.status = `正在规划合成路线…（${seconds}s）`;
          },
        );
        this.synthesisRoutes = data.routes;
        this.status = data.route_count
          ? `找到 ${data.route_count} 条合成路线`
          : "未找到合成路线";
      } catch (err) {
        this.error = err.message;
        this.status = "";
        this.synthesisRoutes = [];
      } finally {
        this.synthesisLoading = false;
      }
    },

    // ---- 标签页 ----
    switchTab(id) {
      if (id === this.activeTabId) return;
      this.activeTabId = id;
      this.pendingAtom = null;
      this.$nextTick(() => this.fitView());
    },
    closeTab(id) {
      const index = this.tabs.findIndex((t) => t.id === id);
      if (index < 0) return;
      const wasActive = id === this.activeTabId;
      const closedSessionId = this.tabs[index].sessionId;
      this.tabs.splice(index, 1);
      if (wasActive) {
        const next = this.tabs[Math.min(index, this.tabs.length - 1)];
        this.activeTabId = next ? next.id : null;
        if (closedSessionId === this.synthesisTargetId) {
          this.synthesisTargetId = next ? next.sessionId : null;
        }
      }
      if (
        closedSessionId === this.synthesisTargetId &&
        !this.tabs.some((tab) => tab.sessionId === this.synthesisTargetId)
      ) {
        this.synthesisTargetId = this.tabs.length ? this.tabs[0].sessionId : null;
      }
      this.synthesisReactantIds = this.synthesisReactantIds.filter(
        (sessionId) => this.tabs.some((tab) => tab.sessionId === sessionId)
      );
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
    async addStructure(op) {
      if (!this.molecule || !this.molecule.atoms.length) {
        this.error = "请先打开或新建分子";
        this.status = "";
        return;
      }
      if (this.pendingAtom === null) {
        this.error = "请先点击一个原子作为锚点";
        this.status = "";
        return;
      }
      const label = op === "add_benzene" ? "苯环" : "硝基";
      const ok = await this.editApi({ op, atom: this.pendingAtom });
      if (ok) this.status = "已加入：" + label;
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
      if (this.editMode === "select") {
        const canvas = this.$refs.canvas;
        const rect = canvas.getBoundingClientRect();
        const atomId = this.hitAtom(event.clientX - rect.left, event.clientY - rect.top);
        this.pendingAtom = atomId;
        this.status = atomId === null ? "未选中原子" : "已选中锚点原子";
        this.render();
        return;
      }
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
      // 新键走 add_bond：π 体系成员与体系外原子可以直接成键；
      // 已有键的键级调整仍走 set_bond_order（涉及 π 体系时后端会拒绝）。
      const ok = await this.editApi({
        op: existing ? "set_bond_order" : "add_bond",
        atom1: atom1,
        atom2: atom2,
        order: this.bondOrder,
      });
      this.pendingAtom = null;
      if (ok) this.status = existing ? "已设置键级" : "已添加键";
    },
    async deleteAtom(atomId) {
      const wasInPi = this.molecule.pi_systems.some((pi) =>
        pi.atoms.includes(atomId)
      );
      const ok = await this.editApi({ op: "del_atom", atom: atomId });
      this.pendingAtom = null;
      if (ok) {
        this.status = wasInPi ? "已删除原子及其 π 体系" : "已删除原子";
      }
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
    // ---- 分子保存 ----
    saveSuggestedName() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab) return "molecule" + MOLECULE_SUFFIX;
      // 已保存过的文件沿用原文件名，新分子用名称或分子式兜底
      const fromPath = (tab.originalPath || "").split(/[\\/]/).pop();
      if (fromPath) {
        // 旧版 .py 文件改按 .mol 建议，避开浏览器下载警告
        return fromPath.replace(/\.py$/i, MOLECULE_SUFFIX);
      }
      return (tab.name || tab.formula || "molecule") + MOLECULE_SUFFIX;
    },
    rejectDisconnectedSave(tab) {
      const count = tab?.molecule?.component_count;
      if (count === 1) return false;
      this.error = count === 0
        ? "空分子不能保存"
        : `分子包含 ${count} 个连通分量，仅支持单一连通分量保存`;
      this.status = "";
      return true;
    },
    async saveMolecule() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab) return;
      if (this.rejectDisconnectedSave(tab)) return;
      this.error = "";
      if (typeof window.showSaveFilePicker !== "function") {
        // 浏览器不提供系统保存对话框时，退回手动输入保存路径
        this.openSaveDialog();
        return;
      }
      this.error = "";
      this.status = "正在保存…";
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: this.saveSuggestedName(),
          types: [
            { description: "分子构建脚本（.mol）", accept: { "text/plain": [MOLECULE_SUFFIX] } },
          ],
        });
        const exported = await this.postJson("/api/export", {
          session_id: tab.sessionId,
          filename: handle.name,
        });
        const writable = await handle.createWritable();
        await writable.write(exported.content);
        await writable.close();
        tab.dirty = false;
        this.status = "已保存到：" + handle.name;
        await this.reloadSavedTab(tab, {
          filename: exported.filename || handle.name,
          content: exported.content,
        });
      } catch (err) {
        if (err && err.name === "AbortError") {
          this.status = "已取消保存"; // 用户在文件管理器中点了取消
          return;
        }
        this.error = "保存失败：" + (err && err.message ? err.message : err);
        this.status = "";
      }
    },
    openSaveDialog() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab) return;
      this.savePath =
        tab.originalPath ||
        "demo_output\\" + (tab.formula || "molecule") + MOLECULE_SUFFIX;
      this.showSaveDialog = true;
    },
    async confirmSave() {
      const tab = this.tabs.find((t) => t.id === this.activeTabId);
      if (!tab) return;
      if (this.rejectDisconnectedSave(tab)) return;
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
        await this.reloadSavedTab(tab, { path: data.path });
      } catch (err) {
        this.error = "请求失败：" + err.message;
        this.status = "";
      }
    },
    async reloadSavedTab(tab, payload) {
      this.status = "正在重新加载…";
      try {
        const response = await fetch("/api/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!data.ok) {
          throw new Error(data.error || "重新加载失败");
        }
        const descriptor = this.loadDescriptor(payload);
        tab.key = descriptor.key;
        tab.source = descriptor.displaySource || data.molecule.source || "";
        tab.name = data.molecule.name || "";
        tab.formula = data.molecule.formula || "";
        tab.molecule = data.molecule;
        tab.sessionId = data.session_id;
        tab.dirty = false;
        // 文件系统访问接口只给出文件名，拿不到完整路径，因此只回填路径来源
        if (typeof payload.path === "string") {
          tab.originalPath = payload.path;
        }
        this.status =
          "已保存并重新加载：" + (payload.path || payload.filename || "");
        this.$nextTick(() => this.fitView());
      } catch (err) {
        this.error = "重新加载失败：" + err.message;
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
        ctx.strokeStyle = this.canvasTheme().pending;
        ctx.lineWidth = Math.max(1.5, 2 * this.zoom);
        ctx.beginPath();
        ctx.arc(px, py, this.atomHitRadius(anchor), 0, Math.PI * 2);
        ctx.stroke();
      }
    },
    drawEmpty(ctx) {
      ctx.fillStyle = this.canvasTheme().empty;
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
      ctx.strokeStyle = this.canvasTheme().bond;
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
      const theme = this.canvasTheme();
      const [x, y] = transform(atom.x, atom.y);
      const color = atom.active
        ? theme.activeH
        : theme.elementColors[atom.element] || theme.elementColors.c;
      const label = atom.label || atom.element.toUpperCase();
      const fontSize = Math.max(4, ATOM_FONT_RATIO * this.zoom);
      ctx.font = "600 " + fontSize + "px 'Segoe UI', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      // 不透明背景椭圆：完全遮住穿过后方的键线（含字母间空隙与边缘）
      const textWidth = ctx.measureText(label).width;
      const pad = fontSize * 0.08;
      ctx.fillStyle = theme.mask;
      ctx.beginPath();
      ctx.ellipse(x, y, textWidth / 2.5 + pad, fontSize / 2.5 + pad, 0, 0, Math.PI * 2);
      ctx.fill();
      // 蒙版描边兜底：防止字形抗锯齿边缘透出键线
      ctx.lineWidth = Math.max(0.3, fontSize * 0.03);
      ctx.strokeStyle = theme.mask;
      ctx.strokeText(label, x, y);
      ctx.fillStyle = color;
      ctx.fillText(label, x, y);
    },
  },
});
window.__viewer = viewerApp.mount("#app");
