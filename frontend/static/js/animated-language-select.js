(() => {
    "use strict";

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const instances = [];
    let nextId = 0;

    function initializeAnimatedSelect(select) {
        if (select.dataset.animatedSelectReady === "true") return;
        select.dataset.animatedSelectReady = "true";
        select.classList.add("animated-language-select__native");
        select.setAttribute("aria-hidden", "true");
        select.tabIndex = -1;

        const instanceId = "animated-language-select-" + (++nextId);
        const root = document.createElement("div");
        root.className = "animated-language-select";

        const trigger = document.createElement("button");
        trigger.type = "button";
        trigger.className = "animated-language-select__trigger";
        trigger.setAttribute("aria-haspopup", "listbox");
        trigger.setAttribute("aria-expanded", "false");
        trigger.setAttribute("aria-controls", instanceId + "-list");

        const value = document.createElement("span");
        value.className = "animated-language-select__value";
        trigger.appendChild(value);

        const popover = document.createElement("div");
        popover.className = "animated-language-select__popover";
        popover.hidden = true;

        const list = document.createElement("div");
        list.id = instanceId + "-list";
        list.className = "animated-language-select__list";
        list.setAttribute("role", "listbox");
        list.setAttribute("aria-label", select.getAttribute("aria-label") || "选择选项");
        list.tabIndex = -1;

        const optionElements = Array.from(select.options).map((option, index) => {
            const item = document.createElement("button");
            item.type = "button";
            item.id = instanceId + "-option-" + index;
            item.className = "animated-language-select__item is-in-view";
            item.setAttribute("role", "option");
            item.setAttribute("data-index", String(index));
            item.tabIndex = -1;
            item.style.setProperty("--animated-item-index", String(index));

            const itemText = document.createElement("span");
            itemText.className = "animated-language-select__item-text";
            itemText.textContent = option.textContent.trim();
            item.appendChild(itemText);
            list.appendChild(item);
            return item;
        });

        const topGradient = document.createElement("div");
        topGradient.className =
            "animated-language-select__gradient animated-language-select__gradient--top";
        topGradient.setAttribute("aria-hidden", "true");

        const bottomGradient = document.createElement("div");
        bottomGradient.className =
            "animated-language-select__gradient animated-language-select__gradient--bottom";
        bottomGradient.setAttribute("aria-hidden", "true");

        popover.append(list, topGradient, bottomGradient);
        root.append(trigger, popover);
        select.insertAdjacentElement("afterend", root);

        let activeIndex = Math.max(0, select.selectedIndex);
        let closeTimer = 0;
        let scrollFrame = 0;

        function syncSelection() {
            const selectedIndex = Math.max(0, select.selectedIndex);
            activeIndex = selectedIndex;
            value.textContent = select.options[selectedIndex]
                ? select.options[selectedIndex].textContent.trim()
                : "请选择";
            optionElements.forEach((item, index) => {
                const selected = index === selectedIndex;
                item.classList.toggle("is-selected", selected);
                item.setAttribute("aria-selected", selected ? "true" : "false");
            });
        }

        function updateGradients() {
            const maximum = Math.max(0, list.scrollHeight - list.clientHeight);
            const topOpacity = Math.min(list.scrollTop / 42, 1);
            const bottomOpacity = maximum === 0
                ? 0
                : Math.min((maximum - list.scrollTop) / 42, 1);
            topGradient.style.opacity = topOpacity.toFixed(3);
            bottomGradient.style.opacity = bottomOpacity.toFixed(3);
        }

        function scheduleViewportUpdate() {
            if (!scrollFrame) {
                scrollFrame = requestAnimationFrame(() => {
                    scrollFrame = 0;
                    if (!popover.hidden) updateGradients();
                });
            }
        }

        function scrollActiveIntoView() {
            const item = optionElements[activeIndex];
            if (!item) return;
            const itemTop = item.offsetTop;
            const itemBottom = itemTop + item.offsetHeight;
            const topBoundary = list.scrollTop + 10;
            const bottomBoundary = list.scrollTop + list.clientHeight - 10;
            let target = null;
            if (itemTop < topBoundary) {
                target = Math.max(0, itemTop - 10);
            } else if (itemBottom > bottomBoundary) {
                target = itemBottom - list.clientHeight + 10;
            }
            if (target !== null) {
                list.scrollTo({
                    top: target,
                    behavior: reducedMotion ? "auto" : "smooth"
                });
            }
        }

        function setActive(index, shouldScroll = false) {
            activeIndex = Math.max(0, Math.min(index, optionElements.length - 1));
            optionElements.forEach((item, itemIndex) => {
                item.classList.toggle("is-active", itemIndex === activeIndex);
            });
            list.setAttribute("aria-activedescendant", optionElements[activeIndex].id);
            if (shouldScroll) scrollActiveIntoView();
        }

        function close(options = {}) {
            const restoreFocus = options.restoreFocus || false;
            const immediate = options.immediate || false;
            if (popover.hidden && !root.classList.contains("is-open")) return;
            window.clearTimeout(closeTimer);
            root.classList.remove("is-open");
            trigger.setAttribute("aria-expanded", "false");
            const finish = () => {
                if (!root.classList.contains("is-open")) popover.hidden = true;
            };
            if (immediate || reducedMotion) {
                finish();
            } else {
                closeTimer = window.setTimeout(finish, 210);
            }
            if (restoreFocus) trigger.focus({ preventScroll: true });
        }

        function closeOthers() {
            instances.forEach((instance) => {
                if (instance.root !== root) instance.close({ immediate: true });
            });
        }

        function open() {
            if (select.disabled || optionElements.length === 0) return;
            closeOthers();
            window.clearTimeout(closeTimer);
            popover.hidden = false;
            root.classList.add("is-open");
            trigger.setAttribute("aria-expanded", "true");
            setActive(Math.max(0, select.selectedIndex));
            requestAnimationFrame(() => {
                scrollActiveIntoView();
                updateGradients();
                list.focus({ preventScroll: true });
            });
        }

        function choose(index) {
            if (!select.options[index]) return;
            select.selectedIndex = index;
            syncSelection();
            select.dispatchEvent(new Event("change", { bubbles: true }));
            close({ restoreFocus: true });
        }

        trigger.addEventListener("click", () => {
            if (root.classList.contains("is-open")) {
                close({ restoreFocus: true });
            } else {
                open();
            }
        });

        trigger.addEventListener("keydown", (event) => {
            if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
                event.preventDefault();
                open();
            }
        });

        list.addEventListener("keydown", (event) => {
            if (event.key === "ArrowDown") {
                event.preventDefault();
                setActive(activeIndex + 1, true);
            } else if (event.key === "ArrowUp") {
                event.preventDefault();
                setActive(activeIndex - 1, true);
            } else if (event.key === "Home") {
                event.preventDefault();
                setActive(0, true);
            } else if (event.key === "End") {
                event.preventDefault();
                setActive(optionElements.length - 1, true);
            } else if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                choose(activeIndex);
            } else if (event.key === "Escape") {
                event.preventDefault();
                close({ restoreFocus: true });
            } else if (event.key === "Tab") {
                close({ immediate: true });
            }
        });

        optionElements.forEach((item, index) => {
            item.addEventListener("pointerenter", () => setActive(index));
            item.addEventListener("click", () => choose(index));
        });
        list.addEventListener("scroll", scheduleViewportUpdate, { passive: true });
        select.addEventListener("change", syncSelection);

        syncSelection();
        instances.push({ root, close, select });
    }

    function initializeWithin(root) {
        const scope = root || document;
        const selects = scope.matches && scope.matches("[data-animated-list-select]")
            ? [scope]
            : Array.from(scope.querySelectorAll("[data-animated-list-select]"));
        selects.forEach((select) => {
            const detail = select.closest("[data-model-version-detail]");
            if (detail && detail.classList.contains("is-hidden")) return;
            initializeAnimatedSelect(select);
        });
    }

    function initializeAll() {
        initializeWithin(document);

        document.addEventListener("pointerdown", (event) => {
            instances.forEach((instance) => {
                if (!instance.root.contains(event.target)) instance.close();
            });
        });

        document.addEventListener("change", (event) => {
            if (event.target.matches("[data-model-version-select]")) {
                instances.forEach((instance) => {
                    if (instance.select !== event.target) {
                        instance.close({ immediate: true });
                    }
                });
            }
        });
        document.addEventListener("model-version-visible", (event) => {
            const detail = event.detail && event.detail.detail;
            if (detail) requestAnimationFrame(() => initializeWithin(detail));
        });
    }

    window.XiezhiAnimatedSelect = { init: initializeWithin };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeAll, { once: true });
    } else {
        initializeAll();
    }
})();
