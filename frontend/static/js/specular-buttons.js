(() => {
    "use strict";

    const PAD = 20;
    const VERTEX_SHADER = `#version 300 es
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}`;

    const FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec2 uCenter;
uniform vec2 uHalfSize;
uniform float uRadius;
uniform float uAngle;
uniform float uPx;
uniform vec3 uLineColor;
uniform vec3 uBaseColor;
uniform float uIntensity;
uniform float uShineSize;
uniform float uShineFade;
uniform float uThickness;
uniform float uBaseWidth;

out vec4 fragColor;

float sdRoundedRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float gaussianLine(float d, float sigma) {
    float x = d / (sigma + 1e-6);
    float k = mix(1.0, 1.6, smoothstep(0.0, 1.5, x));
    return exp(-k * x * x);
}

void main() {
    vec2 p = gl_FragCoord.xy - uCenter;
    float d = sdRoundedRect(p, uHalfSize, uRadius);
    vec2 lightDirection = vec2(cos(uAngle), sin(uAngle));
    float base = (1.0 - smoothstep(0.0, uBaseWidth, abs(d))) * 0.45;
    vec2 ellipticalNormal = normalize(p / (uHalfSize * uHalfSize) + 1e-6);
    float phi = acos(clamp(abs(dot(ellipticalNormal, lightDirection)), 0.0, 1.0));
    float rim = 1.0 - smoothstep(
        uShineSize - uShineFade,
        uShineSize + uShineFade + 1e-4,
        phi
    );
    float line = gaussianLine(d, uThickness);
    float edgeClamp = 1.0 - smoothstep(0.5 * uPx, 3.0 * uPx, abs(d));
    float highlight = line * rim * edgeClamp * uIntensity;
    vec3 color = uBaseColor * base + uLineColor * highlight;
    float alpha = clamp(base + highlight, 0.0, 1.0);
    fragColor = vec4(color, alpha);
}`;

    const CONFIGS = {
        filled: {
            lineColor: "#ffd98a",
            baseColor: "#6f481d",
            intensity: 1.35,
            shineSize: 11,
            shineFade: 38,
            thickness: 1.1,
            proximity: 230
        },
        outline: {
            lineColor: "#fff0c9",
            baseColor: "#b98a4a",
            intensity: 1.18,
            shineSize: 10,
            shineFade: 42,
            thickness: 1,
            proximity: 230
        }
    };

    function compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const message = gl.getShaderInfoLog(shader) || "Unknown shader error";
            gl.deleteShader(shader);
            throw new Error(message);
        }
        return shader;
    }

    function createProgram(gl) {
        const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
        const program = gl.createProgram();
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(program) || "Unknown program link error";
            gl.deleteProgram(program);
            throw new Error(message);
        }
        return program;
    }

    function colorToRgb(hex) {
        const value = hex.replace("#", "");
        const normalized = value.length === 3
            ? value.split("").map(character => character + character).join("")
            : value;
        const number = Number.parseInt(normalized, 16);
        return [
            ((number >> 16) & 255) / 255,
            ((number >> 8) & 255) / 255,
            (number & 255) / 255
        ];
    }

    class SpecularButtonEffect {
        constructor(button) {
            this.button = button;
            this.fx = button.querySelector(".specular-button__fx");
            this.canvas = document.createElement("canvas");
            this.gl = this.canvas.getContext("webgl2", {
                alpha: true,
                premultipliedAlpha: true,
                antialias: true,
                powerPreference: "low-power"
            });
            if (!this.fx || !this.gl) {
                throw new Error("WebGL2 is unavailable");
            }

            this.config = button.classList.contains("specular-button--outline")
                ? CONFIGS.outline
                : CONFIGS.filled;
            this.dpr = Math.min(window.devicePixelRatio || 1, 1.5);
            this.width = 1;
            this.height = 1;
            this.pointerAngle = null;
            this.proximity = 0;
            this.focused = false;
            this.angle = 2.4;
            this.idleAngle = 2.4;
            this.brightness = 0;
            this.lastFrame = performance.now();

            this.program = createProgram(this.gl);
            this.uniforms = {};
            [
                "uCenter",
                "uHalfSize",
                "uRadius",
                "uAngle",
                "uPx",
                "uLineColor",
                "uBaseColor",
                "uIntensity",
                "uShineSize",
                "uShineFade",
                "uThickness",
                "uBaseWidth"
            ].forEach(name => {
                this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
            });

            this.vertexBuffer = this.gl.createBuffer();
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer);
            this.gl.bufferData(
                this.gl.ARRAY_BUFFER,
                new Float32Array([-1, -1, 3, -1, -1, 3]),
                this.gl.STATIC_DRAW
            );

            const position = this.gl.getAttribLocation(this.program, "position");
            this.gl.enableVertexAttribArray(position);
            this.gl.vertexAttribPointer(position, 2, this.gl.FLOAT, false, 0, 0);
            this.gl.clearColor(0, 0, 0, 0);
            this.gl.enable(this.gl.BLEND);
            this.gl.blendFunc(this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA);

            this.onPointerMove = this.handlePointerMove.bind(this);
            this.onFocus = () => {
                this.focused = true;
            };
            this.onBlur = () => {
                this.focused = false;
            };
            window.addEventListener("pointermove", this.onPointerMove, { passive: true });
            button.addEventListener("focus", this.onFocus);
            button.addEventListener("blur", this.onBlur);

            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(button);
            this.fx.appendChild(this.canvas);
            this.resize();
        }

        resize() {
            const rect = this.button.getBoundingClientRect();
            this.width = Math.max(1, rect.width);
            this.height = Math.max(1, rect.height);
            const canvasWidth = Math.round((this.width + PAD * 2) * this.dpr);
            const canvasHeight = Math.round((this.height + PAD * 2) * this.dpr);
            if (this.canvas.width !== canvasWidth || this.canvas.height !== canvasHeight) {
                this.canvas.width = canvasWidth;
                this.canvas.height = canvasHeight;
                this.canvas.style.width = `${this.width + PAD * 2}px`;
                this.canvas.style.height = `${this.height + PAD * 2}px`;
                this.gl.viewport(0, 0, canvasWidth, canvasHeight);
            }
        }

        handlePointerMove(event) {
            const rect = this.button.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            const distanceX = Math.max(rect.left - event.clientX, 0, event.clientX - rect.right);
            const distanceY = Math.max(rect.top - event.clientY, 0, event.clientY - rect.bottom);
            const distance = Math.hypot(distanceX, distanceY);

            if (distance === 0) {
                const normalizedX = (event.clientX - centerX) / Math.max(rect.width / 2, 1);
                const normalizedY = (centerY - event.clientY) / Math.max(rect.height / 2, 1);
                this.pointerAngle =
                    Math.atan2(2 / rect.height, -2 / rect.width) +
                    normalizedX * 0.3 +
                    normalizedY * 0.15;
            } else {
                this.pointerAngle = Math.atan2(centerY - event.clientY, event.clientX - centerX);
            }

            const linear = Math.max(0, 1 - distance / this.config.proximity);
            this.proximity = linear * linear * (3 - 2 * linear);
        }

        update(now) {
            const gl = this.gl;
            const delta = Math.min((now - this.lastFrame) / 1000, 0.05);
            this.lastFrame = now;
            this.idleAngle += 0.24 * delta;

            const shouldFollow = this.pointerAngle !== null && this.proximity > 0;
            const targetAngle = shouldFollow ? this.pointerAngle : this.idleAngle;
            const difference =
                ((targetAngle - this.angle + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
            this.angle += difference * (1 - Math.exp(-delta * 7));

            const brightnessTarget = Math.max(this.proximity, this.focused ? 0.82 : 0);
            this.brightness +=
                (brightnessTarget - this.brightness) * (1 - Math.exp(-delta * 8));

            const lineColor = colorToRgb(this.config.lineColor);
            const baseColor = colorToRgb(this.config.baseColor);
            gl.useProgram(this.program);
            gl.uniform2f(
                this.uniforms.uCenter,
                (PAD + this.width / 2) * this.dpr,
                (PAD + this.height / 2) * this.dpr
            );
            gl.uniform2f(
                this.uniforms.uHalfSize,
                (this.width / 2) * this.dpr,
                (this.height / 2) * this.dpr
            );
            gl.uniform1f(
                this.uniforms.uRadius,
                Math.min(8, Math.min(this.width, this.height) / 2) * this.dpr
            );
            gl.uniform1f(this.uniforms.uAngle, this.angle);
            gl.uniform1f(this.uniforms.uPx, this.dpr);
            gl.uniform3fv(this.uniforms.uLineColor, lineColor);
            gl.uniform3fv(this.uniforms.uBaseColor, baseColor);
            gl.uniform1f(
                this.uniforms.uIntensity,
                this.config.intensity * this.brightness
            );
            gl.uniform1f(
                this.uniforms.uShineSize,
                (this.config.shineSize * Math.PI) / 180
            );
            gl.uniform1f(
                this.uniforms.uShineFade,
                (this.config.shineFade * Math.PI) / 180
            );
            gl.uniform1f(
                this.uniforms.uThickness,
                this.config.thickness * this.dpr
            );
            gl.uniform1f(this.uniforms.uBaseWidth, this.dpr);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
        }

        dispose() {
            window.removeEventListener("pointermove", this.onPointerMove);
            this.button.removeEventListener("focus", this.onFocus);
            this.button.removeEventListener("blur", this.onBlur);
            this.resizeObserver.disconnect();
            if (this.canvas.parentNode === this.fx) {
                this.fx.removeChild(this.canvas);
            }
            this.gl.deleteBuffer(this.vertexBuffer);
            this.gl.deleteProgram(this.program);
            this.gl.getExtension("WEBGL_lose_context")?.loseContext();
        }
    }

    function initializeSpecularButtons() {
        const buttons = [...document.querySelectorAll("[data-specular-button]")];
        if (!buttons.length) return;

        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (reducedMotion) {
            buttons.forEach(button => button.classList.add("is-static"));
            return;
        }

        const effects = [];
        buttons.forEach(button => {
            try {
                effects.push(new SpecularButtonEffect(button));
                button.classList.add("is-specular-ready");
            } catch (error) {
                button.classList.add("is-static");
                console.warn("Specular button effect unavailable.", error);
            }
        });
        if (!effects.length) return;

        let animationFrame = 0;
        const render = now => {
            effects.forEach(effect => effect.update(now));
            animationFrame = requestAnimationFrame(render);
        };
        const start = () => {
            if (!animationFrame && document.visibilityState === "visible") {
                animationFrame = requestAnimationFrame(render);
            }
        };
        const pause = () => {
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
                animationFrame = 0;
            }
        };
        const handleVisibility = () => {
            if (document.visibilityState === "visible") {
                effects.forEach(effect => {
                    effect.lastFrame = performance.now();
                });
                start();
            } else {
                pause();
            }
        };
        const cleanup = () => {
            pause();
            document.removeEventListener("visibilitychange", handleVisibility);
            effects.forEach(effect => effect.dispose());
        };

        document.addEventListener("visibilitychange", handleVisibility);
        window.addEventListener("pagehide", cleanup, { once: true });
        start();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeSpecularButtons, {
            once: true
        });
    } else {
        initializeSpecularButtons();
    }
})();
