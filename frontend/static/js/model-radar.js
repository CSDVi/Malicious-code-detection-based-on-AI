(function () {
    "use strict";

    const SVG_NS = "http://www.w3.org/2000/svg";
    const CENTER_X = 210;
    const CENTER_Y = 160;
    const RADIUS = 92;
    const LABEL_RADIUS = 127;
    const METRICS = [
        { key: "accuracy", label: "准确率", abbreviation: "ACC" },
        { key: "precision", label: "精确率", abbreviation: "PRE" },
        { key: "f1", label: "F1 分数", abbreviation: "F1" },
        { key: "sensitivity", label: "敏感度", abbreviation: "TPR" },
        { key: "specificity", label: "特异度", abbreviation: "TNR" }
    ];
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function svgElement(name, attributes) {
        const element = document.createElementNS(SVG_NS, name);
        Object.entries(attributes || {}).forEach(function (entry) {
            element.setAttribute(entry[0], String(entry[1]));
        });
        return element;
    }

    function point(index, scale) {
        const angle = -Math.PI / 2 + index * (Math.PI * 2 / METRICS.length);
        return {
            x: CENTER_X + Math.cos(angle) * RADIUS * scale,
            y: CENTER_Y + Math.sin(angle) * RADIUS * scale
        };
    }

    function pointsFor(values) {
        return values.map(function (value, index) {
            const target = point(index, value);
            return target.x.toFixed(2) + "," + target.y.toFixed(2);
        }).join(" ");
    }

    function parseNumber(source, key) {
        const raw = source && source.dataset ? source.dataset[key] : "";
        if (raw === "" || raw === undefined) return null;
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
    }

    function clampRate(value) {
        return value === null ? null : Math.min(1, Math.max(0, value));
    }

    function selectedSource(panel) {
        const detail = panel.closest("[data-model-version-detail]");
        const select = detail ? detail.querySelector("[data-version-language-select]") : null;
        const option = select && select.selectedIndex >= 0 ? select.options[select.selectedIndex] : null;
        return {
            source: option || panel,
            scope: option && option.dataset.languageLabel
                ? option.dataset.languageLabel
                : panel.dataset.defaultScope
        };
    }

    function readData(panel) {
        const selected = selectedSource(panel);
        const source = selected.source;
        const accuracy = clampRate(parseNumber(source, "accuracy"));
        const precision = clampRate(parseNumber(source, "precision"));
        const f1 = clampRate(parseNumber(source, "f1"));
        const falsePositiveRate = clampRate(parseNumber(source, "falsePositiveRate"));
        const falseNegativeRate = clampRate(parseNumber(source, "falseNegativeRate"));
        const counts = {
            tn: parseNumber(source, "tn"),
            fp: parseNumber(source, "fp"),
            fn: parseNumber(source, "fn"),
            tp: parseNumber(source, "tp")
        };
        return {
            scope: selected.scope,
            values: [
                accuracy,
                precision,
                f1,
                falseNegativeRate === null ? null : 1 - falseNegativeRate,
                falsePositiveRate === null ? null : 1 - falsePositiveRate
            ],
            counts: counts
        };
    }

    function appendMetricLabel(svg, metric, index) {
        const angle = -Math.PI / 2 + index * (Math.PI * 2 / METRICS.length);
        const x = CENTER_X + Math.cos(angle) * LABEL_RADIUS;
        const y = CENTER_Y + Math.sin(angle) * LABEL_RADIUS;
        const anchor = Math.abs(x - CENTER_X) < 12 ? "middle" : x < CENTER_X ? "end" : "start";
        const label = svgElement("text", {
            x: x,
            y: y - 5,
            "text-anchor": anchor,
            class: "model-radar-label"
        });
        const name = svgElement("tspan", { x: x, dy: 0 });
        name.textContent = metric.label + " (" + metric.abbreviation + ")";
        const value = svgElement("tspan", { x: x, dy: 14, class: "model-radar-value", "data-radar-value": index });
        value.textContent = "—";
        label.appendChild(name);
        label.appendChild(value);
        svg.appendChild(label);
        return value;
    }

    function buildChart(panel) {
        if (panel._radar) return;
        const svg = panel.querySelector("[data-radar-svg]");
        if (!svg) return;

        [0.2, 0.4, 0.6, 0.8, 1].forEach(function (level) {
            svg.appendChild(svgElement("polygon", {
                points: pointsFor(METRICS.map(function () { return level; })),
                class: "model-radar-grid" + (level === 1 ? " is-major" : "")
            }));
            const tick = svgElement("text", {
                x: CENTER_X + 5,
                y: CENTER_Y - RADIUS * level + 3,
                class: "model-radar-tick"
            });
            tick.textContent = level.toFixed(1);
            svg.appendChild(tick);
        });

        const valueLabels = [];
        METRICS.forEach(function (metric, index) {
            const outer = point(index, 1);
            svg.appendChild(svgElement("line", {
                x1: CENTER_X,
                y1: CENTER_Y,
                x2: outer.x,
                y2: outer.y,
                class: "model-radar-axis"
            }));
            valueLabels.push(appendMetricLabel(svg, metric, index));
        });

        const shape = svgElement("polygon", { class: "model-radar-shape" });
        svg.appendChild(shape);
        const dots = METRICS.map(function () {
            const dot = svgElement("circle", { r: 3.2, class: "model-radar-dot" });
            svg.appendChild(dot);
            return dot;
        });
        const empty = svgElement("text", {
            x: CENTER_X,
            y: CENTER_Y,
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            class: "model-radar-empty"
        });
        empty.textContent = "暂无完整指标";
        svg.appendChild(empty);

        panel._radar = {
            svg: svg,
            shape: shape,
            dots: dots,
            valueLabels: valueLabels,
            empty: empty,
            values: null,
            frame: 0
        };
        updateEvaluation(panel, false);
    }

    function paintRadar(panel, values) {
        const chart = panel._radar;
        chart.shape.setAttribute("points", pointsFor(values));
        values.forEach(function (value, index) {
            const target = point(index, value);
            chart.dots[index].setAttribute("cx", target.x.toFixed(2));
            chart.dots[index].setAttribute("cy", target.y.toFixed(2));
            chart.valueLabels[index].textContent = (value * 100).toFixed(1) + "%";
        });
    }

    function updateRadar(panel, data, animate) {
        const chart = panel._radar;
        const complete = data.values.every(function (value) { return value !== null; });
        cancelAnimationFrame(chart.frame);
        chart.shape.style.display = complete ? "" : "none";
        chart.empty.style.display = complete ? "none" : "";
        chart.dots.forEach(function (dot) { dot.style.display = complete ? "" : "none"; });
        chart.valueLabels.forEach(function (label) {
            if (!complete) label.textContent = "—";
        });
        if (!complete) {
            chart.values = null;
            chart.svg.setAttribute("aria-label", (data.scope || "当前版本") + " 暂无完整的五项指标");
            return;
        }

        chart.svg.setAttribute(
            "aria-label",
            (data.scope || "当前版本") + "：" + METRICS.map(function (metric, index) {
                return metric.label + " " + (data.values[index] * 100).toFixed(1) + "%";
            }).join("，")
        );
        const startValues = chart.values;
        chart.values = data.values.slice();
        if (!animate || reduceMotion || !startValues) {
            paintRadar(panel, data.values);
            return;
        }

        const startedAt = performance.now();
        const duration = 320;
        const tick = function (now) {
            const progress = Math.min(1, (now - startedAt) / duration);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = data.values.map(function (value, index) {
                return startValues[index] + (value - startValues[index]) * eased;
            });
            paintRadar(panel, current);
            if (progress < 1) chart.frame = requestAnimationFrame(tick);
        };
        chart.frame = requestAnimationFrame(tick);
    }

    function updateConfusionMatrix(panel, data) {
        const matrix = panel.querySelector(".model-confusion-matrix");
        const empty = panel.querySelector("[data-confusion-empty]");
        const keys = ["tn", "fp", "fn", "tp"];
        const complete = keys.every(function (key) {
            return Number.isFinite(data.counts[key]) && data.counts[key] >= 0;
        });
        matrix.hidden = !complete;
        empty.hidden = complete;
        if (!complete) return;

        const normalTotal = data.counts.tn + data.counts.fp;
        const maliciousTotal = data.counts.fn + data.counts.tp;
        const descriptions = {
            tn: "正确放行",
            fp: "误报",
            fn: "漏报",
            tp: "正确拦截"
        };
        keys.forEach(function (key) {
            const count = data.counts[key];
            const denominator = key === "tn" || key === "fp" ? normalTotal : maliciousTotal;
            const rate = denominator ? count / denominator : 0;
            const cell = panel.querySelector('[data-confusion-cell="' + key + '"]');
            panel.querySelector('[data-confusion-count="' + key + '"]').textContent = count.toLocaleString("zh-CN");
            panel.querySelector('[data-confusion-rate="' + key + '"]').textContent = (rate * 100).toFixed(1) + "% · " + descriptions[key];
            cell.style.setProperty("--matrix-intensity", (0.12 + rate * 0.72).toFixed(3));
            cell.setAttribute("aria-label", key.toUpperCase() + " " + count + "，" + descriptions[key] + " " + (rate * 100).toFixed(1) + "%");
        });
    }

    function updateEvaluation(panel, animate) {
        const data = readData(panel);
        updateRadar(panel, data, animate);
        updateConfusionMatrix(panel, data);
    }

    function initializeWithin(root) {
        const scope = root || document;
        const panels = scope.matches && scope.matches("[data-model-radar]")
            ? [scope]
            : Array.from(scope.querySelectorAll("[data-model-radar]"));
        panels.forEach((panel) => {
            const detail = panel.closest("[data-model-version-detail]");
            if (!detail || !detail.classList.contains("is-hidden")) buildChart(panel);
        });
    }

    initializeWithin(document);

    document.addEventListener("model-version-visible", function (event) {
        const detail = event.detail && event.detail.detail;
        if (detail) requestAnimationFrame(function () { initializeWithin(detail); });
    });

    document.addEventListener("change", function (event) {
        const select = event.target.closest && event.target.closest("[data-version-language-select]");
        if (!select) return;
        const detail = select.closest("[data-model-version-detail]");
        const panel = detail ? detail.querySelector("[data-model-radar]") : null;
        if (!panel) return;
        if (!panel._radar) buildChart(panel);
        requestAnimationFrame(function () {
            updateEvaluation(panel, false);
        });
    });
})();
