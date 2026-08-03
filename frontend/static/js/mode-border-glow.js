(function () {
    const precisePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!precisePointer || reducedMotion) return;

    document.querySelectorAll("[data-mode-border-glow]").forEach(function (card) {
        if (card.classList.contains("disabled")) return;

        let frame = 0;
        let pointerX = 0;
        let pointerY = 0;

        function render() {
            frame = 0;
            const rect = card.getBoundingClientRect();
            const x = pointerX - rect.left;
            const y = pointerY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const dx = x - centerX;
            const dy = y - centerY;
            const edgeX = dx === 0 ? 0 : Math.abs(dx) / centerX;
            const edgeY = dy === 0 ? 0 : Math.abs(dy) / centerY;
            const edgeProximity = Math.min(1, Math.max(edgeX, edgeY));
            let angle = Math.atan2(dy, dx) * (180 / Math.PI) + 90;
            if (angle < 0) angle += 360;

            card.style.setProperty("--edge-proximity", (edgeProximity * 100).toFixed(3));
            card.style.setProperty("--cursor-angle", angle.toFixed(3) + "deg");
        }

        card.addEventListener("pointermove", function (event) {
            pointerX = event.clientX;
            pointerY = event.clientY;
            if (!frame) frame = window.requestAnimationFrame(render);
        }, { passive: true });

        card.addEventListener("pointerleave", function () {
            if (frame) {
                window.cancelAnimationFrame(frame);
                frame = 0;
            }
            card.style.setProperty("--edge-proximity", "0");
        });
    });
})();
