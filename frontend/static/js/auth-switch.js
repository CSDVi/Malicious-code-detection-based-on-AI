(function () {
    const shell = document.querySelector("[data-auth-shell]");
    if (!shell) return;

    const forms = Array.from(shell.querySelectorAll("[data-auth-form]"));
    const switchCopies = Array.from(shell.querySelectorAll("[data-auth-copy]"));
    const switchLinks = Array.from(shell.querySelectorAll("[data-auth-switch]"));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let currentMode = shell.dataset.authInitialMode === "register" ? "register" : "login";
    let focusTimer = 0;

    function applyMode(mode, options = {}) {
        const nextMode = mode === "register" ? "register" : "login";
        currentMode = nextMode;
        shell.classList.toggle("is-register", nextMode === "register");
        shell.dataset.authMode = nextMode;

        forms.forEach((panel) => {
            const active = panel.dataset.authForm === nextMode;
            panel.setAttribute("aria-hidden", String(!active));
            panel.inert = !active;
        });
        switchCopies.forEach((copy) => {
            const active = copy.dataset.authCopy === nextMode;
            copy.setAttribute("aria-hidden", String(!active));
            copy.inert = !active;
        });

        document.title = nextMode === "register"
            ? "注册 - 獬豸安码"
            : "登录 - 獬豸安码";

        if (options.updateHistory) {
            const target = switchLinks.find((link) => link.dataset.authSwitch === nextMode);
            if (target && window.location.pathname !== target.pathname) {
                window.history.pushState({ authMode: nextMode }, "", target.href);
            }
            document.querySelectorAll(".flash-error").forEach((message) => {
                message.hidden = true;
            });
        }

        window.clearTimeout(focusTimer);
        if (options.focus) {
            focusTimer = window.setTimeout(() => {
                const activePanel = forms.find((panel) => panel.dataset.authForm === nextMode);
                const input = activePanel && activePanel.querySelector("input");
                if (input) input.focus({ preventScroll: true });
            }, reduceMotion ? 0 : 1080);
        }
    }

    switchLinks.forEach((link) => {
        link.addEventListener("click", (event) => {
            if (
                event.defaultPrevented
                || event.button !== 0
                || event.metaKey
                || event.ctrlKey
                || event.shiftKey
                || event.altKey
            ) {
                return;
            }
            event.preventDefault();
            applyMode(link.dataset.authSwitch, { updateHistory: true, focus: true });
        });
    });

    window.addEventListener("popstate", () => {
        const mode = window.location.pathname.endsWith("/register") ? "register" : "login";
        applyMode(mode, { focus: false });
    });

    applyMode(currentMode);
    window.requestAnimationFrame(() => shell.classList.add("is-ready"));
})();
