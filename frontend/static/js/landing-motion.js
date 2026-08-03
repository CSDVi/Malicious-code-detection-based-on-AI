(function () {
    const root = document.querySelector("[data-landing-motion]");
    if (!root || root.dataset.motionInitialized === "true") return;
    root.dataset.motionInitialized = "true";

    const curtain = root.querySelector("[data-opening-curtain]");
    const openingEnabled = root.dataset.openingEnabled === "true" && Boolean(curtain);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function unlockPage() {
        document.body.classList.remove("is-opening");
    }

    function showStaticPage() {
        unlockPage();
        root.classList.remove("is-motion-active");
        root.classList.add("is-motion-static");
        if (curtain) curtain.hidden = true;
    }

    if (!openingEnabled || reduceMotion || !window.gsap) {
        showStaticPage();
        return;
    }

    const { gsap } = window;
    const openingWord = root.querySelector("[data-opening-word]");
    const openingRule = root.querySelector("[data-opening-rule]");
    const openingLockup = root.querySelector(".opening-lockup");
    const nav = document.querySelector(".module-cards");
    const titleParts = gsap.utils.toArray("[data-opening-title]", root);
    const heroVisual = root.querySelector("[data-opening-visual]");
    const contentGroups = gsap.utils.toArray("[data-opening-content]", root);
    const isCompact = window.matchMedia("(max-width: 900px)").matches;

    root.classList.add("is-motion-active");
    document.body.classList.add("is-opening");

    try {
        gsap.set(openingWord, {
            yPercent: 132,
            scaleX: 0.62,
            transformOrigin: "50% 100%"
        });
        gsap.set(openingRule, { scaleX: 0, transformOrigin: "0% 50%" });
        if (nav) gsap.set(nav, { yPercent: -110, autoAlpha: 0 });
        gsap.set(titleParts, {
            yPercent: 128,
            scaleX: 0.58,
            transformOrigin: "50% 100%"
        });
        gsap.set(heroVisual, {
            xPercent: isCompact ? 0 : 8,
            scale: 1.06,
            clipPath: "inset(0 0 0 100%)"
        });
        gsap.set(contentGroups, {
            y: isCompact ? 38 : 58,
            autoAlpha: 0
        });

        const opening = gsap.timeline({
            defaults: { ease: "power4.out" },
            onComplete: () => {
                unlockPage();
                gsap.set(curtain, { display: "none" });
                gsap.set(titleParts, { clearProps: "transform" });
                gsap.set(heroVisual, { clearProps: "transform,clipPath" });
                gsap.set(contentGroups, { clearProps: "transform,opacity,visibility" });
                if (nav) gsap.set(nav, { clearProps: "transform,opacity,visibility" });
            }
        });

        opening
            .to(openingWord, {
                yPercent: 0,
                scaleX: 1,
                duration: isCompact ? 1.05 : 1.35
            }, 0.18)
            .to(openingRule, {
                scaleX: 1,
                duration: 1.1,
                ease: "power3.inOut"
            }, 0.42)
            .to(openingWord, {
                letterSpacing: isCompact ? "0.08em" : "0.14em",
                duration: 0.8,
                ease: "power2.inOut"
            }, 1.15)
            .to(openingLockup, {
                y: -32,
                autoAlpha: 0,
                duration: 0.58,
                ease: "power3.in"
            }, 1.92)
            .to(curtain, {
                clipPath: "inset(0 0 100% 0)",
                duration: 1.2,
                ease: "power4.inOut"
            }, 2.2);

        if (nav) {
            opening.to(nav, {
                yPercent: 0,
                autoAlpha: 1,
                duration: 1.05
            }, 2.48);
        }

        opening
            .to(heroVisual, {
                xPercent: 0,
                scale: 1,
                clipPath: "inset(0 0 0 0%)",
                duration: 1.45
            }, 2.38)
            .to(titleParts, {
                yPercent: 0,
                scaleX: 1,
                duration: 1.35,
                stagger: 0.13
            }, 2.68)
            .to(contentGroups, {
                y: 0,
                autoAlpha: 1,
                duration: 1.05,
                stagger: 0.12,
                ease: "power3.out"
            }, 3.08);
    } catch (_error) {
        gsap.set(
            [...titleParts, heroVisual, ...contentGroups, nav].filter(Boolean),
            { clearProps: "all" }
        );
        showStaticPage();
    }
})();
