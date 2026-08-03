(function () {
    const precisePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!precisePointer || reducedMotion) return;

    document.querySelectorAll("[data-proximity-list]").forEach(function (list) {
        const cards = Array.from(list.querySelectorAll("[data-proximity-card]"));
        if (!cards.length) return;

        const states = cards.map(function (card) {
            return { card: card, lift: 0, renderedLift: 0, targetLift: 0, intensity: 0 };
        });
        let frame = 0;
        let pointerInside = false;
        let pointerX = 0;
        let pointerY = 0;

        function setTargets() {
            states.forEach(function (state) {
                if (!pointerInside) {
                    state.targetLift = 0;
                    state.intensity = 0;
                    return;
                }

                const rect = state.card.getBoundingClientRect();
                const baseTop = rect.top - state.renderedLift;
                const baseBottom = baseTop + rect.height;
                const dx = Math.max(rect.left - pointerX, 0, pointerX - rect.right);
                const dy = Math.max(baseTop - pointerY, 0, pointerY - baseBottom);
                const distance = Math.hypot(dx, dy);
                const linear = Math.max(0, 1 - distance / 150);
                const intensity = linear * linear * (3 - 2 * linear);

                state.intensity = intensity;
                state.targetLift = -7 * intensity;
            });
        }

        function animate() {
            let unsettled = false;
            states.forEach(function (state) {
                state.lift += (state.targetLift - state.lift) * 0.2;
                if (Math.abs(state.targetLift - state.lift) > 0.02) {
                    unsettled = true;
                }

                const shadowStrength = Math.max(state.intensity, Math.min(1, Math.abs(state.lift) / 7));
                state.renderedLift = Math.round(state.lift);
                state.card.style.transform = state.renderedLift
                    ? "translateY(" + state.renderedLift + "px)"
                    : "none";
                state.card.style.filter = "none";
                state.card.style.boxShadow = shadowStrength > 0.01
                    ? "0 " + (7 + shadowStrength * 7).toFixed(1) + "px " + (12 + shadowStrength * 10).toFixed(1) + "px rgba(0,0,0," + (0.18 + shadowStrength * 0.17).toFixed(3) + ")"
                    : "none";
                state.card.classList.toggle("is-proximity-active", shadowStrength > 0.025);
            });

            frame = unsettled ? window.requestAnimationFrame(animate) : 0;
        }

        function schedule() {
            setTargets();
            if (!frame) frame = window.requestAnimationFrame(animate);
        }

        list.addEventListener("pointermove", function (event) {
            pointerInside = true;
            pointerX = event.clientX;
            pointerY = event.clientY;
            schedule();
        }, { passive: true });

        list.addEventListener("pointerleave", function () {
            pointerInside = false;
            schedule();
        });
    });
})();
