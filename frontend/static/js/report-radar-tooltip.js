(function () {
    "use strict";

    let tooltip = null;

    function ensureTooltip() {
        if (tooltip) return tooltip;
        tooltip = document.createElement("div");
        tooltip.className = "report-radar-tooltip";
        tooltip.hidden = true;
        tooltip.setAttribute("role", "tooltip");
        document.body.appendChild(tooltip);
        return tooltip;
    }

    function positionTooltip(event, target) {
        const box = ensureTooltip();
        const rect = target.getBoundingClientRect();
        const x = event && Number.isFinite(event.clientX) ? event.clientX : rect.left + rect.width / 2;
        const y = event && Number.isFinite(event.clientY) ? event.clientY : rect.top;
        const left = Math.min(window.innerWidth - box.offsetWidth - 10, Math.max(10, x + 14));
        const top = Math.min(window.innerHeight - box.offsetHeight - 10, Math.max(10, y + 14));
        box.style.left = left + "px";
        box.style.top = top + "px";
    }

    function show(target, event) {
        const svg = target.closest("svg");
        if (!svg) return;
        const index = target.dataset.radarAxisIndex;
        const values = Array.from(svg.querySelectorAll('[data-radar-series-dot][data-axis-index="' + index + '"]'));
        if (!values.length) return;
        const title = target.dataset.axisLabel + " (" + target.dataset.axisAbbreviation + ")";
        const rows = values.map(function (dot) {
            return '<div class="report-radar-tooltip-row"><span><i class="' + dot.dataset.styleKey + '"></i>'
                + dot.dataset.modelName + '</span><b>' + dot.dataset.value + '%</b></div>';
        }).join("");
        const box = ensureTooltip();
        box.innerHTML = "<strong>" + title + "</strong>" + rows;
        box.hidden = false;
        positionTooltip(event, target);
    }

    function hide() {
        if (tooltip) tooltip.hidden = true;
    }

    document.addEventListener("pointerover", function (event) {
        const target = event.target.closest && event.target.closest("[data-radar-axis-index]");
        if (target) show(target, event);
    });
    document.addEventListener("pointermove", function (event) {
        const target = event.target.closest && event.target.closest("[data-radar-axis-index]");
        if (target && tooltip && !tooltip.hidden) positionTooltip(event, target);
    });
    document.addEventListener("pointerout", function (event) {
        if (event.target.closest && event.target.closest("[data-radar-axis-index]")) hide();
    });
    document.addEventListener("focusin", function (event) {
        if (event.target.matches && event.target.matches("[data-radar-axis-index]")) show(event.target, null);
    });
    document.addEventListener("focusout", function (event) {
        if (event.target.matches && event.target.matches("[data-radar-axis-index]")) hide();
    });
})();
