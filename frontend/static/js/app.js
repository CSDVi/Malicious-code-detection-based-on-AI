(function () {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function setupPageTransitions() {
        const stage = document.querySelector("[data-page-transition-stage]");
        if (!stage || reduceMotion) return;

        let leaving = false;
        const transitionLinks = document.querySelectorAll(
            ".module-card, .content-tab, [data-page-transition]"
        );

        transitionLinks.forEach((link) => {
            link.addEventListener("click", (event) => {
                if (
                    leaving
                    || event.defaultPrevented
                    || event.button !== 0
                    || event.metaKey
                    || event.ctrlKey
                    || event.shiftKey
                    || event.altKey
                    || link.hasAttribute("download")
                    || link.target
                ) {
                    return;
                }

                const destination = new URL(link.href, window.location.href);
                const current = new URL(window.location.href);
                const isSameDocumentHash = (
                    destination.origin === current.origin
                    && destination.pathname === current.pathname
                    && destination.search === current.search
                    && destination.hash
                    && destination.hash !== current.hash
                );
                const isCurrentPage = (
                    destination.origin === current.origin
                    && destination.pathname === current.pathname
                    && destination.search === current.search
                    && destination.hash === current.hash
                );

                if (
                    destination.origin !== current.origin
                    || !["http:", "https:"].includes(destination.protocol)
                    || isSameDocumentHash
                    || isCurrentPage
                ) {
                    return;
                }

                event.preventDefault();
                leaving = true;
                link.classList.add("is-switching");
                stage.classList.add("is-leaving");
                stage.setAttribute("aria-busy", "true");
                window.setTimeout(() => {
                    window.location.assign(destination.href);
                }, 220);
            });
        });

        window.addEventListener("pageshow", () => {
            leaving = false;
            stage.classList.remove("is-leaving");
            stage.removeAttribute("aria-busy");
            transitionLinks.forEach((link) => link.classList.remove("is-switching"));
        });
    }

    setupPageTransitions();

    function riskClass(level) {
        if (!level) return "risk-low";
        const normalized = level.toLowerCase();
        if (level.includes("严重") || normalized.includes("critical")) return "risk-critical";
        if (level.includes("高危") || normalized.includes("high")) return "risk-high";
        if (level.includes("中危") || normalized.includes("medium")) return "risk-mid";
        if (level.includes("安全") || normalized.includes("safe")) return "risk-safe";
        return "risk-low";
    }

    document.querySelectorAll("[data-risk-level]:not([data-page-row])").forEach((el) => {
        el.classList.add(riskClass(el.dataset.riskLevel));
    });

    document.querySelectorAll(".upload-zone input[type='file']").forEach((input) => {
        const updateFileState = () => {
            const zone = input.closest(".upload-zone");
            const label = zone && zone.querySelector("[data-file-label]");
            if (!zone || !label) return;
            const file = input.files[0];
            const maximum = Number(input.dataset.maxBytes || 0);
            if (file && maximum > 0 && file.size > maximum) {
                input.value = "";
                zone.classList.remove("is-ready");
                label.textContent = `文件超过 ${Math.round(maximum / (1024 ** 3))} GB 上限`;
                return;
            }
            zone.classList.toggle("is-ready", input.files.length > 0);
            label.textContent = file ? file.name : "等待选择文件";
            if (window.gsap && !reduceMotion) {
                gsap.fromTo(zone, { scale: 0.99 }, { scale: 1, duration: 0.35, ease: "power2.out" });
            }
        };
        input.addEventListener("change", updateFileState);
        const zone = input.closest(".upload-zone");
        if (!zone) return;
        ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => {
            event.preventDefault();
            event.stopPropagation();
            zone.classList.add("is-dragover");
            if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        }));
        ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => {
            event.preventDefault();
            event.stopPropagation();
            zone.classList.remove("is-dragover");
        }));
        zone.addEventListener("drop", (event) => {
            const file = event.dataTransfer && event.dataTransfer.files[0];
            if (!file) return;
            const accepted = (input.getAttribute("accept") || "")
                .split(",")
                .map((value) => value.trim().toLowerCase())
                .filter(Boolean);
            const extension = file.name.includes(".") ? `.${file.name.split(".").pop().toLowerCase()}` : "";
            const label = zone.querySelector("[data-file-label]");
            if (accepted.length && !accepted.includes(extension)) {
                zone.classList.remove("is-ready");
                if (label) label.textContent = `不支持 ${extension || "该"} 文件`;
                return;
            }
            const transfer = new DataTransfer();
            transfer.items.add(file);
            input.files = transfer.files;
            updateFileState();
        });
    });

    const tray = document.getElementById("scan-job-tray");
    if (tray) {
        const activeJobCacheKey = "xiezhi_active_scan_jobs";
        const list = tray.querySelector("[data-job-list]");
        const jobsUrl = tray.dataset.jobsUrl;
        const fileUrl = tray.dataset.fileUrl;
        const projectUrl = tray.dataset.projectUrl;
        const activeStatuses = new Set(["queued", "running", "cancelling"]);
        const terminalStatuses = new Set(["completed", "failed", "cancelled"]);
        const seenActiveJobs = new Set();
        const dismissedJobs = new Set();
        let latestJobs = [];
        const statusText = {
            queued: "等待执行", running: "正在检测", cancelling: "正在停止",
            cancelled: "已停止", completed: "检测完成", failed: "执行失败"
        };
        const modeText = {
            auto: "自动模式", quick: "快速模式", standard: "标准模式", deep: "深度模式"
        };
        const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (character) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
        })[character]);
        const specularJobButton = ({ label, variant = "neutral", attributes = "", disabled = false }) => `
            <button type="button"
                    class="compact-button job-specular-button job-specular-button--${variant}"
                    data-job-specular
                    ${attributes}${disabled ? " disabled aria-disabled=\"true\"" : ""}>
                <span class="job-specular-button__label" data-job-specular-label>${escapeHtml(label)}</span>
            </button>`;
        const setSpecularJobButtonLabel = (button, label) => {
            const labelNode = button.querySelector("[data-job-specular-label]");
            if (labelNode) {
                labelNode.textContent = label;
            } else {
                button.textContent = label;
            }
        };
        let lastSpecularPointer = null;
        let specularPointerFrame = 0;
        const updateSpecularJobButtons = () => {
            specularPointerFrame = 0;
            if (!lastSpecularPointer || tray.hidden || !list) return;
            list.querySelectorAll("[data-job-specular]").forEach((button) => {
                const rect = button.getBoundingClientRect();
                if (!rect.width || !rect.height) return;
                const distanceX = Math.max(
                    rect.left - lastSpecularPointer.x,
                    0,
                    lastSpecularPointer.x - rect.right
                );
                const distanceY = Math.max(
                    rect.top - lastSpecularPointer.y,
                    0,
                    lastSpecularPointer.y - rect.bottom
                );
                const distance = Math.hypot(distanceX, distanceY);
                const linear = Math.max(0, 1 - distance / 180);
                const strength = linear * linear * (3 - 2 * linear);
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                const angle = Math.atan2(
                    lastSpecularPointer.y - centerY,
                    lastSpecularPointer.x - centerX
                ) * 180 / Math.PI + 90;
                const localX = Math.max(0, Math.min(100,
                    (lastSpecularPointer.x - rect.left) / rect.width * 100
                ));
                const localY = Math.max(0, Math.min(100,
                    (lastSpecularPointer.y - rect.top) / rect.height * 100
                ));
                button.style.setProperty("--job-specular-angle", `${angle.toFixed(2)}deg`);
                button.style.setProperty("--job-specular-strength", strength.toFixed(3));
                button.style.setProperty("--job-specular-x", `${localX.toFixed(2)}%`);
                button.style.setProperty("--job-specular-y", `${localY.toFixed(2)}%`);
            });
        };
        if (!reduceMotion) {
            document.addEventListener("pointermove", (event) => {
                lastSpecularPointer = { x: event.clientX, y: event.clientY };
                if (!specularPointerFrame) {
                    specularPointerFrame = requestAnimationFrame(updateSpecularJobButtons);
                }
            }, { passive: true });
            window.addEventListener("blur", () => {
                if (!list) return;
                list.querySelectorAll("[data-job-specular]").forEach((button) => {
                    button.style.setProperty("--job-specular-strength", "0");
                });
            });
        }
        const renderJobs = (jobs) => {
            latestJobs = jobs;
            jobs
                .filter((job) => activeStatuses.has(job.status))
                .forEach((job) => seenActiveJobs.add(String(job.id)));
            const visible = jobs.filter((job) => {
                const jobId = String(job.id);
                if (dismissedJobs.has(jobId)) return false;
                return activeStatuses.has(job.status)
                    || (seenActiveJobs.has(jobId) && terminalStatuses.has(job.status));
            });
            try {
                sessionStorage.setItem(
                    activeJobCacheKey,
                    JSON.stringify(
                        jobs.filter((job) => activeStatuses.has(job.status))
                    )
                );
            } catch (_error) {}
            tray.hidden = visible.length === 0;
            if (!list) return;
            if (!visible.length) {
                tray.dataset.visibleJobIds = "";
                list.innerHTML = "";
                return;
            }
            tray.dataset.visibleJobIds = visible.map((job) => job.id).join(",");
            list.innerHTML = visible.map((job) => {
                const jobId = String(job.id);
                const total = Number(job.total_files || 0);
                const done = Number(job.processed_files || 0);
                const percent = total > 0 ? Math.min(100, Math.round(done * 100 / total)) : 0;
                const isFileJob = job.target_type === "file";
                const progress = isFileJob && total > 0
                    ? `${percent}%`
                    : total > 0 ? `${done}/${total}` : "准备中";
                const resultUrl = isFileJob ? fileUrl : projectUrl;
                const action = job.pending_upload
                    ? `<button type="button" class="secondary-action compact-button" disabled>正在提交</button>`
                    : job.status === "queued" || job.status === "running"
                    ? specularJobButton({
                        label: "停止",
                        variant: "danger",
                        attributes: `data-cancel-job="${escapeHtml(jobId)}"`
                    })
                    : job.status === "cancelling"
                        ? specularJobButton({
                            label: "正在停止",
                            variant: "danger",
                            disabled: true
                        })
                    : job.status === "completed"
                        ? `<a class="secondary-action compact-button" href="${resultUrl}?job_id=${encodeURIComponent(jobId)}">查看结果</a>`
                        : "";
                const dismiss = `<button class="job-row-dismiss" type="button"
                    data-dismiss-job="${escapeHtml(jobId)}"
                    aria-label="关闭此任务提示" title="关闭"><span aria-hidden="true">×</span></button>`;
                return `<div class="scan-job-row" data-global-job="${jobId}">
                    <div class="scan-job-main"><strong>${escapeHtml(job.target_name || jobId.slice(0, 12))}</strong><span>${escapeHtml(job.stage || statusText[job.status] || job.status)} · ${escapeHtml(modeText[job.mode] || job.mode || "未记录模式")}</span></div>
                    <div class="scan-job-progress" role="progressbar" aria-label="检测进度 ${progress}"
                         aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}">
                        <span style="width:${percent}%"></span>
                    </div>
                    <span class="scan-job-count">${progress}</span>
                    <div class="scan-job-actions">${action}${dismiss}</div>
                </div>`;
            }).join("");
        };
        try {
            const cachedJobs = JSON.parse(
                sessionStorage.getItem(activeJobCacheKey) || "[]"
            );
            if (Array.isArray(cachedJobs) && cachedJobs.length) {
                renderJobs(cachedJobs);
            }
        } catch (_error) {}

        document.querySelectorAll("[data-project-scan-form], [data-file-scan-form]").forEach((form) => {
            form.addEventListener("submit", () => {
                const isFileScan = form.hasAttribute("data-file-scan-form");
                const upload = form.querySelector(
                    isFileScan ? "input[name='code_file']" : "input[name='project_zip']"
                );
                if (!upload || !upload.files || !upload.files[0]) return;
                const modeInput = form.querySelector(
                    "input[name='mode']:checked, select[name='mode']"
                );
                const pendingJob = {
                    id: `upload-${Date.now()}`,
                    target_name: upload.files[0].name,
                    target_type: isFileScan ? "file" : "project",
                    mode: modeInput ? modeInput.value : "auto",
                    status: "queued",
                    stage: isFileScan ? "正在上传文件并创建检测任务" : "正在上传项目并创建检测任务",
                    processed_files: isFileScan ? 5 : 0,
                    total_files: isFileScan ? 100 : 0,
                    pending_upload: true
                };
                renderJobs([
                    pendingJob,
                    ...latestJobs.filter((job) => !job.pending_upload)
                ]);
                const submit = form.querySelector("button[type='submit']");
                if (submit) {
                    submit.disabled = true;
                    submit.textContent = "正在提交";
                }
            });
        });
        const pollJobs = (scheduleNext = true) => fetch(jobsUrl, {headers: {"Accept": "application/json"}})
            .then((response) => response.ok ? response.json() : Promise.reject(new Error("job list unavailable")))
            .then((data) => renderJobs(data.jobs || []))
            .catch(() => {})
            .finally(() => { if (scheduleNext) window.setTimeout(pollJobs, 1500); });
        document.addEventListener("click", (event) => {
            const dismissButton = event.target.closest("[data-dismiss-job]");
            if (dismissButton) {
                dismissedJobs.add(String(dismissButton.dataset.dismissJob));
                renderJobs(latestJobs);
                return;
            }
            const button = event.target.closest("[data-cancel-job]");
            if (!button) return;
            const jobId = String(button.dataset.cancelJob);
            document.querySelectorAll(`[data-cancel-job="${CSS.escape(jobId)}"]`).forEach((candidate) => {
                candidate.disabled = true;
                candidate.setAttribute("aria-disabled", "true");
                setSpecularJobButtonLabel(candidate, "正在停止");
            });
            fetch(`${jobsUrl}/${encodeURIComponent(jobId)}/cancel`, {
                method: "POST", headers: {"Accept": "application/json"}
            }).then((response) => {
                if (!response.ok) throw new Error("cancel request failed");
                return pollJobs(false);
            }).catch(() => {
                document.querySelectorAll(`[data-cancel-job="${CSS.escape(jobId)}"]`).forEach((candidate) => {
                    candidate.disabled = false;
                    candidate.removeAttribute("aria-disabled");
                    setSpecularJobButtonLabel(candidate, "停止");
                });
            });
        });
        pollJobs();
    }

    function setupInteractiveRiskDonuts() {
        document.querySelectorAll("[data-risk-donut]").forEach((donut) => {
            const svg = donut.querySelector("svg");
            const segments = Array.from(donut.querySelectorAll("[data-donut-segment]"));
            const proxies = Array.from(donut.querySelectorAll("[data-risk-donut-proxy]"));
            if (!svg || !segments.length) return;

            let selectedKey = "";
            const setActive = (key) => {
                [...segments, ...proxies].forEach((item) => {
                    item.classList.toggle("is-active", Boolean(key) && item.dataset.riskFilter === key);
                });
            };
            const setSelected = (key) => {
                selectedKey = key;
                [...segments, ...proxies].forEach((item) => {
                    item.classList.toggle("is-selected", Boolean(key) && item.dataset.riskFilter === key);
                });
            };
            const segmentForPercent = (percent) => segments.find((segment) => {
                const start = Number(segment.dataset.startPercent || 0);
                const end = Number(segment.dataset.endPercent || 0);
                return percent >= start && percent <= end;
            }) || segments[segments.length - 1];
            const activate = (key) => {
                if (!key) return;
                setSelected(key);
                setActive(key);
                document.dispatchEvent(new CustomEvent("xiezhi:project-risk-filter", {
                    detail: { riskLevel: key }
                }));
            };

            donut.addEventListener("pointermove", (event) => {
                const rect = svg.getBoundingClientRect();
                if (!rect.width || !rect.height) return;
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                const dx = event.clientX - centerX;
                const dy = event.clientY - centerY;
                const distance = Math.hypot(dx, dy);
                const radius = Math.min(rect.width, rect.height) / 2;
                if (distance > radius * 1.08 || distance < radius * 0.32) {
                    setActive(selectedKey);
                    return;
                }
                const angle = (Math.atan2(dy, dx) * 180 / Math.PI + 90 + 360) % 360;
                const segment = segmentForPercent(angle / 3.6);
                setActive(segment ? segment.dataset.riskFilter : selectedKey);
            }, { passive: true });
            donut.addEventListener("pointerleave", () => setActive(selectedKey));

            [...segments, ...proxies].forEach((item) => {
                const key = item.dataset.riskFilter;
                item.addEventListener("pointerenter", () => setActive(key));
                item.addEventListener("focus", () => setActive(key));
                item.addEventListener("blur", () => setActive(selectedKey));
                item.addEventListener("click", () => activate(key));
                item.addEventListener("keydown", (event) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    activate(key);
                });
            });
        });
    }

    setupInteractiveRiskDonuts();

    document.querySelectorAll("[data-paginated-table]").forEach((container) => {
        const rows = Array.from(container.querySelectorAll("[data-page-row]"));
        const pagination = container.querySelector("[data-table-pagination]");
        const firstPage = pagination && pagination.querySelector("[data-page-first]");
        const previous = pagination && pagination.querySelector("[data-page-prev]");
        const next = pagination && pagination.querySelector("[data-page-next]");
        const lastPage = pagination && pagination.querySelector("[data-page-last]");
        const status = pagination && pagination.querySelector("[data-page-status]");
        const pageInput = pagination && pagination.querySelector("[data-page-input]");
        const pageGo = pagination && pagination.querySelector("[data-page-go]");
        const searchInput = pagination && pagination.querySelector("[data-table-search]");
        const extensionAll = pagination && pagination.querySelector("[data-extension-all]");
        const extensionOptions = pagination
            ? Array.from(pagination.querySelectorAll("[data-extension-option]"))
            : [];
        const extensionSummary = pagination && pagination.querySelector("[data-extension-summary]");
        const filterKinds = pagination
            ? Array.from(pagination.querySelectorAll("[data-filter-kind]"))
            : [];
        const filterOptions = pagination
            ? Array.from(pagination.querySelectorAll("[data-filter-option]"))
            : [];
        const filterAllControls = pagination
            ? Array.from(pagination.querySelectorAll("[data-filter-all]"))
            : [];
        const filterBranches = pagination
            ? Array.from(pagination.querySelectorAll("[data-filter-branch]"))
            : [];
        const filterSummary = pagination && pagination.querySelector("[data-filter-summary]");
        const emptyRow = container.querySelector("[data-filter-empty]");
        const pageSize = Math.max(1, Number(container.dataset.pageSize || 10));
        const hasTreeFilter = filterKinds.length > 0 && filterOptions.length > 0;
        let activeFilterKind = (
            filterKinds.find((option) => option.checked)
            || filterKinds[0]
            || {}
        ).value || "";
        let panelHighlightTimer = 0;
        let pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
        let currentPage = 1;

        if (!pagination || !previous || !next || !status || rows.length === 0) return;

        const filteredRows = () => {
            const query = String(searchInput ? searchInput.value : "").trim().toLowerCase();
            const serialQuery = /^\d+$/.test(query);
            const selectedExtensions = new Set(
                extensionOptions.filter((option) => option.checked).map((option) => option.value)
            );
            const selectedTreeValues = new Set(
                optionsForKind(activeFilterKind)
                    .filter((option) => option.checked)
                    .map((option) => option.value)
            );
            return rows.filter((row) => {
                const matchesQuery = !query || (
                    serialQuery
                        ? row.dataset.projectSerial === query
                        : String(row.dataset.fileSearch || row.textContent || "").toLowerCase().includes(query)
                );
                const matchesExtension = !extensionOptions.length
                    || selectedExtensions.has(String(row.dataset.fileExtension || "__none__"));
                const matchesTreeFilter = !hasTreeFilter
                    || selectedTreeValues.has(treeFilterValue(row, activeFilterKind));
                return matchesQuery && matchesExtension && matchesTreeFilter;
            });
        };

        const optionsForKind = (kind) => filterOptions.filter((option) => {
            return option.dataset.filterGroup === kind;
        });
        const filterLabel = (kind) => kind === "risk" ? "危险等级" : "语言";
        const treeFilterValue = (row, kind) => {
            if (kind === "risk") return String(row.dataset.riskLevel || "unknown");
            return String(row.dataset.fileLanguage || "unknown");
        };
        const updateTreeFilterSummary = () => {
            if (!hasTreeFilter) return;
            activeFilterKind = (
                filterKinds.find((option) => option.checked)
                || filterKinds[0]
                || {}
            ).value || activeFilterKind;
            filterBranches.forEach((branch) => {
                branch.hidden = branch.dataset.filterBranch !== activeFilterKind;
            });
            filterAllControls.forEach((control) => {
                const groupOptions = optionsForKind(control.dataset.filterAll);
                const selectedCount = groupOptions.filter((option) => option.checked).length;
                control.checked = groupOptions.length > 0 && selectedCount === groupOptions.length;
                control.indeterminate = selectedCount > 0 && selectedCount < groupOptions.length;
            });
            if (!filterSummary) return;
            const activeOptions = optionsForKind(activeFilterKind);
            const selectedOptions = activeOptions.filter((option) => option.checked);
            const label = filterLabel(activeFilterKind);
            filterSummary.textContent = selectedOptions.length === activeOptions.length
                ? `${label}：全部`
                : selectedOptions.length === 0
                    ? `${label}：未选择`
                    : selectedOptions.length === 1
                        ? `${label}：${selectedOptions[0].dataset.filterLabel || selectedOptions[0].value}`
                        : `${label}：${selectedOptions.length} 项`;
        };

        const updateExtensionSummary = () => {
            if (!extensionOptions.length) return;
            const selectedCount = extensionOptions.filter((option) => option.checked).length;
            if (extensionAll) {
                extensionAll.checked = selectedCount === extensionOptions.length;
                extensionAll.indeterminate = selectedCount > 0 && selectedCount < extensionOptions.length;
            }
            if (extensionSummary) {
                extensionSummary.textContent = selectedCount === extensionOptions.length
                    ? "全部"
                    : selectedCount === 0 ? "未选择" : `${selectedCount} 类`;
            }
        };

        const renderPage = () => {
            const visibleRows = filteredRows();
            pageCount = Math.max(1, Math.ceil(visibleRows.length / pageSize));
            currentPage = Math.min(currentPage, pageCount);
            const first = (currentPage - 1) * pageSize;
            const last = first + pageSize;
            const visibleSet = new Set(visibleRows.slice(first, last));
            rows.forEach((row) => {
                row.hidden = !visibleSet.has(row);
            });
            if (emptyRow) emptyRow.hidden = visibleRows.length !== 0;
            pagination.hidden = !pagination.hasAttribute("data-always-visible") && pageCount <= 1;
            status.textContent = visibleRows.length
                ? `第 ${currentPage} / ${pageCount} 页 · 共 ${visibleRows.length} 项`
                : "共 0 项";
            if (pageInput) {
                pageInput.max = String(pageCount);
                pageInput.value = String(currentPage);
                pageInput.disabled = visibleRows.length === 0;
            }
            if (pageGo) pageGo.disabled = visibleRows.length === 0;
            if (firstPage) firstPage.disabled = visibleRows.length === 0 || currentPage === 1;
            previous.disabled = visibleRows.length === 0 || currentPage === 1;
            next.disabled = visibleRows.length === 0 || currentPage === pageCount;
            if (lastPage) lastPage.disabled = visibleRows.length === 0 || currentPage === pageCount;
            updateExtensionSummary();
            updateTreeFilterSummary();
        };

        if (firstPage) firstPage.addEventListener("click", () => {
            currentPage = 1;
            renderPage();
        });
        previous.addEventListener("click", () => {
            if (currentPage === 1) return;
            currentPage -= 1;
            renderPage();
        });
        next.addEventListener("click", () => {
            if (currentPage === pageCount) return;
            currentPage += 1;
            renderPage();
        });
        if (lastPage) lastPage.addEventListener("click", () => {
            currentPage = pageCount;
            renderPage();
        });
        const jumpToPage = () => {
            if (!pageInput) return;
            const requested = Number.parseInt(pageInput.value, 10);
            if (!Number.isFinite(requested)) {
                pageInput.value = String(currentPage);
                return;
            }
            currentPage = Math.min(pageCount, Math.max(1, requested));
            renderPage();
        };
        if (pageGo) pageGo.addEventListener("click", jumpToPage);
        if (pageInput) {
            pageInput.addEventListener("change", jumpToPage);
            pageInput.addEventListener("keydown", (event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                jumpToPage();
            });
        }
        if (searchInput) searchInput.addEventListener("input", () => {
            currentPage = 1;
            renderPage();
        });
        if (extensionAll) extensionAll.addEventListener("change", () => {
            extensionOptions.forEach((option) => {
                option.checked = extensionAll.checked;
            });
            currentPage = 1;
            renderPage();
        });
        extensionOptions.forEach((option) => {
            option.addEventListener("change", () => {
                currentPage = 1;
                renderPage();
            });
        });
        filterKinds.forEach((option) => {
            option.addEventListener("change", () => {
                activeFilterKind = option.value;
                currentPage = 1;
                renderPage();
            });
        });
        filterAllControls.forEach((control) => {
            control.addEventListener("change", () => {
                optionsForKind(control.dataset.filterAll).forEach((option) => {
                    option.checked = control.checked;
                });
                currentPage = 1;
                renderPage();
            });
        });
        filterOptions.forEach((option) => {
            option.addEventListener("change", () => {
                currentPage = 1;
                renderPage();
            });
        });
        document.addEventListener("xiezhi:project-risk-filter", (event) => {
            const riskLevel = String((event.detail || {}).riskLevel || "");
            const riskKind = filterKinds.find((option) => option.value === "risk");
            const riskOptions = optionsForKind("risk");
            const target = riskOptions.find((option) => option.value === riskLevel);
            if (!riskKind || !target) return;
            riskKind.checked = true;
            activeFilterKind = "risk";
            riskOptions.forEach((option) => {
                option.checked = option === target;
            });
            currentPage = 1;
            renderPage();
            container.classList.add("is-filter-target");
            window.clearTimeout(panelHighlightTimer);
            panelHighlightTimer = window.setTimeout(() => {
                container.classList.remove("is-filter-target");
            }, 1400);
            container.scrollIntoView({
                behavior: reduceMotion ? "auto" : "smooth",
                block: "start"
            });
        });
        renderPage();
    });

    if (!window.gsap || reduceMotion) {
        return;
    }

    gsap.defaults({ duration: 0.58, ease: "power3.out" });
    const tl = gsap.timeline();
    const introCards = document.querySelectorAll(".motion-card:not(.top-nav), .metric-card, .panel, .command-panel, .table-panel");
    if (introCards.length) {
        tl.from(introCards, {
            y: 18,
            autoAlpha: 0,
            stagger: { each: 0.045, from: "start" },
            overwrite: "auto"
        }, "<0.1");
    }

    document.querySelectorAll("[data-count]").forEach((el) => {
        const raw = Number(el.dataset.count);
        if (!Number.isFinite(raw)) return;
        const decimals = raw % 1 === 0 ? 0 : 1;
        const state = { value: 0 };
        gsap.to(state, {
            value: raw,
            duration: 0.9,
            ease: "power2.out",
            onUpdate: () => {
                el.textContent = state.value.toFixed(decimals);
            }
        });
    });

    document.querySelectorAll(".module-card, .primary-action, .secondary-action, .danger-action").forEach((el) => {
        el.addEventListener("mouseenter", () => gsap.to(el, { y: -1, duration: 0.18, overwrite: "auto" }));
        el.addEventListener("mouseleave", () => gsap.to(el, { y: 0, duration: 0.18, overwrite: "auto" }));
    });
})();
