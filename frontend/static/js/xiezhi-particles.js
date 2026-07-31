import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const host = document.querySelector("[data-particle-scene]");
const canvas = document.getElementById("xiezhi-particle-canvas");

if (!host || !canvas) {
    throw new Error("Particle scene mount point is missing");
}

const loading = host.querySelector("[data-particle-loading]");
const loadingProgress = host.querySelector("[data-particle-progress]");
const errorState = host.querySelector("[data-particle-error]");
const particleCountLabel = host.querySelector("[data-particle-count]");
const pointerRipple = host.querySelector("[data-particle-ripple]");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const MIN_PARTICLE_COUNT = 6000;
const MAX_PARTICLE_COUNT = 24000;
const INITIAL_PARTICLE_COUNT = 6000;
const deviceMemory = Number(navigator.deviceMemory || 8);
const processorCount = Number(navigator.hardwareConcurrency || 8);
const lowPowerDevice = window.innerWidth < 720 || deviceMemory <= 4 || processorCount <= 4;
const modelUrl = host.dataset.modelUrl || "/static/models/xiezhi.glb";
const particleDataUrl = host.dataset.particleDataUrl || "";
const modelRevision = host.dataset.modelRevision || "v1";
const targetPointCount = MAX_PARTICLE_COUNT;
const cacheKey = `${modelUrl}|${modelRevision}|${targetPointCount}`;
const idleFrameInterval = 1000 / (reduceMotion ? 15 : 30);
const interactiveFrameInterval = 1000 / (reduceMotion ? 20 : lowPowerDevice ? 30 : 60);

let renderer;
try {
    renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: false,
        desynchronized: true,
        powerPreference: "high-performance"
    });
} catch (webglError) {
    console.error("WebGL is unavailable", webglError);
    showError("当前浏览器无法创建 WebGL 粒子场");
    throw webglError;
}

renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
const hardwareParticleBudget = chooseRenderPointCount(renderer);
host.dataset.initialParticleBudget = String(INITIAL_PARTICLE_COUNT);
host.dataset.hardwareParticleBudget = String(hardwareParticleBudget);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
camera.position.set(0, 0.08, 6.7);

const root = new THREE.Group();
const particleGroup = new THREE.Group();
const fieldGroup = new THREE.Group();
root.add(particleGroup, fieldGroup);
scene.add(root);

const pointer = new THREE.Vector2();
const pointerTarget = new THREE.Vector2();
const clock = new THREE.Clock();
const baseRootPosition = new THREE.Vector3(0.08, -0.08, 0);
const baseYaw = -0.08;
const animatedMaterials = [];
let pointerPresence = 0;
let pointerPresenceTarget = 0;
let isIntersecting = true;
let pageVisible = !document.hidden;
let renderRunning = false;
let destroyed = false;
let animationFrame = 0;
let lastFrameTime = 0;
let pointerBounds = null;
let pendingPointerPosition = null;
let pointerUpdateFrame = 0;
let particleGeometry = null;
let currentDrawCount = 0;
let slowInteractiveFrames = 0;
let fastInteractiveFrames = 0;
let maximumAdaptiveDrawCount = INITIAL_PARTICLE_COUNT;
let startupBudgetTimer = 0;

function chooseRenderPointCount(webglRenderer) {
    const memoryFactor = THREE.MathUtils.clamp(deviceMemory / 8, 0.25, 1.25);
    const processorFactor = THREE.MathUtils.clamp(processorCount / 8, 0.25, 2);
    const viewportMegapixels = Math.max(0.45, window.innerWidth * window.innerHeight / 1000000);
    const resolutionPenalty = Math.sqrt(viewportMegapixels / 2.07);
    let gpuFactor = 1;

    try {
        const context = webglRenderer.getContext();
        const debugInfo = context.getExtension("WEBGL_debug_renderer_info");
        const gpuName = debugInfo
            ? String(context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || "")
            : "";
        if (/swiftshader|llvmpipe|software|microsoft basic/i.test(gpuName)) gpuFactor = 0.45;
        else if (context.getParameter(context.MAX_TEXTURE_SIZE) < 8192) gpuFactor = 0.72;
        host.dataset.gpuClass = gpuName || "webgl-default";
    } catch (_error) {
        gpuFactor = 0.85;
    }

    const hardwareIndex = Math.sqrt(memoryFactor * processorFactor) * gpuFactor;
    const mobileFactor = window.innerWidth < 680 ? 0.78 : 1;
    const rawBudget = 13000 * hardwareIndex * mobileFactor / Math.max(0.72, resolutionPenalty);
    return Math.round(THREE.MathUtils.clamp(rawBudget, MIN_PARTICLE_COUNT, MAX_PARTICLE_COUNT) / 500) * 500;
}

function updateVisibleParticleCount(nextCount) {
    if (!particleGeometry) return;
    const availableCount = particleGeometry.attributes.position?.count || INITIAL_PARTICLE_COUNT;
    currentDrawCount = Math.round(
        THREE.MathUtils.clamp(nextCount, MIN_PARTICLE_COUNT, Math.min(MAX_PARTICLE_COUNT, availableCount)) / 500
    ) * 500;
    particleGeometry.setDrawRange(0, currentDrawCount);
    if (particleCountLabel) particleCountLabel.textContent = currentDrawCount.toLocaleString("zh-CN");
    host.dataset.currentParticleCount = String(currentDrawCount);
}

function applyHardwareParticleBudget() {
    startupBudgetTimer = 0;
    if (destroyed || !particleGeometry) return;
    if (pointerPresenceTarget > 0) {
        startupBudgetTimer = window.setTimeout(applyHardwareParticleBudget, 350);
        return;
    }
    maximumAdaptiveDrawCount = Math.min(
        hardwareParticleBudget,
        particleGeometry.attributes.position?.count || hardwareParticleBudget
    );
    updateVisibleParticleCount(maximumAdaptiveDrawCount);
    host.dataset.particleBudgetSource = "hardware";
}

function scheduleHardwareParticleBudget() {
    window.clearTimeout(startupBudgetTimer);
    host.dataset.particleBudgetSource = "initial-default";
    startupBudgetTimer = window.setTimeout(applyHardwareParticleBudget, 800);
}

function setLoading(visible) {
    if (loading) loading.hidden = !visible;
}

function setLoadingMessage(message) {
    if (loadingProgress) loadingProgress.textContent = message;
}

function showError(message = "本地模型未能加载，检测功能不受影响") {
    setLoading(false);
    if (errorState) {
        const label = errorState.querySelector("span");
        if (label) label.textContent = message;
        errorState.hidden = false;
    }
}

function markReady(source) {
    setLoading(false);
    host.classList.add("is-ready");
    host.dataset.particleSource = source;
}

function updateLoadProgress(event) {
    if (!loadingProgress) return;
    const loadedMB = event.loaded / 1024 / 1024;
    if (event.total > 0) {
        const totalMB = event.total / 1024 / 1024;
        const percent = Math.min(100, Math.round(event.loaded / event.total * 100));
        loadingProgress.textContent = `${percent}% · ${loadedMB.toFixed(1)} / ${totalMB.toFixed(1)} MB`;
    } else {
        loadingProgress.textContent = `已载入 ${loadedMB.toFixed(1)} MB`;
    }
}

function createField() {
    const count = lowPowerDevice ? 150 : 260;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const seeds = new Float32Array(count);

    for (let i = 0; i < count; i += 1) {
        const radius = 2.1 + Math.random() * 2.7;
        const angle = Math.random() * Math.PI * 2;
        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] = (Math.random() - 0.46) * 3.2;
        positions[i * 3 + 2] = Math.sin(angle) * radius * 0.42 - 0.45;
        sizes[i] = 0.7 + Math.random() * 1.7;
        seeds[i] = Math.random();
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));

    const material = new THREE.ShaderMaterial({
        precision: "mediump",
        uniforms: {
            uTime: { value: 0 },
            uPixelRatio: { value: renderer.getPixelRatio() }
        },
        vertexShader: `
            uniform float uTime;
            uniform float uPixelRatio;
            attribute float aSize;
            attribute float aSeed;
            varying float vAlpha;
            void main() {
                vec3 p = position;
                p.y += sin(uTime * 0.22 + aSeed * 18.0) * 0.035;
                p.x += cos(uTime * 0.15 + aSeed * 23.0) * 0.018;
                vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
                gl_PointSize = clamp(aSize * uPixelRatio * (5.5 / max(1.0, -mvPosition.z)), 0.8, 3.0);
                gl_Position = projectionMatrix * mvPosition;
                vAlpha = 0.10 + aSeed * 0.16;
            }
        `,
        fragmentShader: `
            varying float vAlpha;
            void main() {
                float d = distance(gl_PointCoord, vec2(0.5));
                float alpha = smoothstep(0.5, 0.04, d) * vAlpha;
                gl_FragColor = vec4(0.84, 0.62, 0.24, alpha);
            }
        `,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });

    animatedMaterials.push(material);
    fieldGroup.add(new THREE.Points(geometry, material));
}

function createRings() {
    const ringColors = [0x8d6428, 0xd4a64f, 0xf0ce82];
    [1.28, 1.55, 1.88].forEach((radius, index) => {
        const points = Array.from({ length: 128 }, (_, i) => {
            const angle = i / 128 * Math.PI * 2;
            return new THREE.Vector3(Math.cos(angle) * radius, -1.58, Math.sin(angle) * radius * 0.28);
        });
        const material = new THREE.LineBasicMaterial({
            color: ringColors[index],
            transparent: true,
            opacity: index === 1 ? 0.28 : 0.15,
            blending: THREE.AdditiveBlending
        });
        root.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), material));
    });

    const ticks = [];
    for (let i = 0; i < 32; i += 1) {
        const angle = i / 32 * Math.PI * 2;
        const inner = i % 4 === 0 ? 1.9 : 1.98;
        const outer = i % 4 === 0 ? 2.1 : 2.05;
        ticks.push(
            Math.cos(angle) * inner, -1.575, Math.sin(angle) * inner * 0.28,
            Math.cos(angle) * outer, -1.575, Math.sin(angle) * outer * 0.28
        );
    }
    const tickGeometry = new THREE.BufferGeometry();
    tickGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
    root.add(new THREE.LineSegments(
        tickGeometry,
        new THREE.LineBasicMaterial({ color: 0xb98c3e, transparent: true, opacity: 0.16 })
    ));
}

function triangleIndices(indexAttribute, triangle) {
    const offset = triangle * 3;
    return indexAttribute
        ? [indexAttribute.getX(offset), indexAttribute.getX(offset + 1), indexAttribute.getX(offset + 2)]
        : [offset, offset + 1, offset + 2];
}

function buildSurfaceData(model) {
    const surfaces = [];
    const va = new THREE.Vector3();
    const vb = new THREE.Vector3();
    const vc = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();

    model.updateMatrixWorld(true);
    model.traverse((object) => {
        if (!object.isMesh || !object.geometry?.attributes?.position) return;

        const position = object.geometry.attributes.position;
        const normal = object.geometry.attributes.normal || null;
        const index = object.geometry.index;
        const triangleCount = index ? index.count / 3 : position.count / 3;
        const distribution = new Float32Array(triangleCount);
        let totalArea = 0;

        for (let triangle = 0; triangle < triangleCount; triangle += 1) {
            const [ia, ib, ic] = triangleIndices(index, triangle);
            va.fromBufferAttribute(position, ia).applyMatrix4(object.matrixWorld);
            vb.fromBufferAttribute(position, ib).applyMatrix4(object.matrixWorld);
            vc.fromBufferAttribute(position, ic).applyMatrix4(object.matrixWorld);
            ab.subVectors(vb, va);
            ac.subVectors(vc, va);
            totalArea += ab.cross(ac).length() * 0.5;
            distribution[triangle] = totalArea;
        }

        if (totalArea > 0) {
            surfaces.push({ mesh: object, position, normal, index, distribution, totalArea });
        }
    });

    if (!surfaces.length) throw new Error("No renderable mesh surface found in GLB model");
    return surfaces;
}

function lowerBound(values, target) {
    let low = 0;
    let high = values.length - 1;
    while (low < high) {
        const middle = (low + high) >>> 1;
        if (values[middle] < target) low = middle + 1;
        else high = middle;
    }
    return low;
}

function yieldToBrowser() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function sampleModelSurface(model) {
    const surfaces = buildSurfaceData(model);
    const totalArea = surfaces.reduce((sum, surface) => sum + surface.totalArea, 0);
    const surfaceDistribution = new Float32Array(surfaces.length);
    let accumulatedArea = 0;
    surfaces.forEach((surface, index) => {
        accumulatedArea += surface.totalArea;
        surfaceDistribution[index] = accumulatedArea;
    });

    const bounds = new THREE.Box3().setFromObject(model);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const normalizedScale = 3.25 / Math.max(size.x, size.y, size.z, 0.001);
    const positions = new Float32Array(targetPointCount * 3);
    const normals = new Float32Array(targetPointCount * 3);
    const colors = new Float32Array(targetPointCount * 3);
    const sizes = new Float32Array(targetPointCount);
    const seeds = new Float32Array(targetPointCount);

    const va = new THREE.Vector3();
    const vb = new THREE.Vector3();
    const vc = new THREE.Vector3();
    const na = new THREE.Vector3();
    const nb = new THREE.Vector3();
    const nc = new THREE.Vector3();
    const sampledPosition = new THREE.Vector3();
    const sampledNormal = new THREE.Vector3();
    const faceNormal = new THREE.Vector3();
    const edgeA = new THREE.Vector3();
    const edgeB = new THREE.Vector3();
    const bronze = new THREE.Color("#7b4f17");
    const gold = new THREE.Color("#d3a348");
    const champagne = new THREE.Color("#ffe7a3");
    const hotGold = new THREE.Color("#ffc85a");
    const color = new THREE.Color();

    for (let pointIndex = 0; pointIndex < targetPointCount; pointIndex += 1) {
        if (pointIndex > 0 && pointIndex % 3000 === 0) {
            setLoadingMessage(`生成粒子 ${Math.round(pointIndex / targetPointCount * 100)}%`);
            await yieldToBrowser();
            if (destroyed) throw new Error("Particle scene was disposed");
        }

        const surfaceIndex = lowerBound(surfaceDistribution, Math.random() * totalArea);
        const surface = surfaces[surfaceIndex];
        const triangle = lowerBound(surface.distribution, Math.random() * surface.totalArea);
        const [ia, ib, ic] = triangleIndices(surface.index, triangle);
        const sqrtR1 = Math.sqrt(Math.random());
        const weightA = 1 - sqrtR1;
        const weightB = sqrtR1 * (1 - Math.random());
        const weightC = 1 - weightA - weightB;

        va.fromBufferAttribute(surface.position, ia);
        vb.fromBufferAttribute(surface.position, ib);
        vc.fromBufferAttribute(surface.position, ic);
        sampledPosition.set(0, 0, 0)
            .addScaledVector(va, weightA)
            .addScaledVector(vb, weightB)
            .addScaledVector(vc, weightC)
            .applyMatrix4(surface.mesh.matrixWorld)
            .sub(center)
            .multiplyScalar(normalizedScale);

        if (surface.normal) {
            na.fromBufferAttribute(surface.normal, ia);
            nb.fromBufferAttribute(surface.normal, ib);
            nc.fromBufferAttribute(surface.normal, ic);
            sampledNormal.set(0, 0, 0)
                .addScaledVector(na, weightA)
                .addScaledVector(nb, weightB)
                .addScaledVector(nc, weightC)
                .transformDirection(surface.mesh.matrixWorld)
                .normalize();
        } else {
            edgeA.subVectors(vb, va);
            edgeB.subVectors(vc, va);
            faceNormal.crossVectors(edgeA, edgeB).transformDirection(surface.mesh.matrixWorld).normalize();
            sampledNormal.copy(faceNormal);
        }

        const offset = pointIndex * 3;
        positions[offset] = sampledPosition.x;
        positions[offset + 1] = sampledPosition.y - 0.02;
        positions[offset + 2] = sampledPosition.z;
        normals[offset] = sampledNormal.x;
        normals[offset + 1] = sampledNormal.y;
        normals[offset + 2] = sampledNormal.z;

        const height = THREE.MathUtils.clamp((sampledPosition.y + 1.62) / 3.24, 0, 1);
        const facing = Math.max(0, sampledNormal.z);
        color.copy(bronze).lerp(gold, height * 0.58 + facing * 0.24);
        if (Math.random() < 0.065) color.lerp(champagne, 0.7);
        if (Math.random() < 0.018) color.copy(hotGold);
        colors[offset] = color.r;
        colors[offset + 1] = color.g;
        colors[offset + 2] = color.b;
        sizes[pointIndex] = 0.9 + Math.pow(Math.random(), 2.15) * 2.0;
        seeds[pointIndex] = Math.random();
    }

    return { positions, normals, colors, sizes, seeds, count: targetPointCount };
}

function createParticleMaterial() {
    const material = new THREE.ShaderMaterial({
        precision: "mediump",
        uniforms: {
            uTime: { value: 0 },
            uPixelRatio: { value: renderer.getPixelRatio() },
            uPointer: { value: new THREE.Vector2() },
            uPointerStrength: { value: 0 },
            uAspect: { value: 1 }
        },
        vertexShader: `
            uniform float uTime;
            uniform float uPixelRatio;
            uniform vec2 uPointer;
            uniform float uPointerStrength;
            uniform float uAspect;
            attribute vec3 aNormal;
            attribute vec3 aColor;
            attribute float aSize;
            attribute float aSeed;
            varying vec3 vColor;
            varying float vTwinkle;
            varying float vInteraction;
            void main() {
                float breath = sin(uTime * 0.56 + position.y * 1.24 + aSeed * 6.28318) * 0.013;
                vec3 p = position + aNormal * breath;

                vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
                vec4 clipPosition = projectionMatrix * mvPosition;
                vec2 delta = clipPosition.xy / max(0.001, clipPosition.w) - uPointer;
                vec2 metricDelta = vec2(delta.x * uAspect, delta.y);
                float pointerDistance = length(metricDelta);
                float grain = fract(aSeed * 17.371);
                float reach = 0.105 + fract(aSeed * 5.731) * 0.05;
                float repel = 1.0 - smoothstep(0.022, reach, pointerDistance);
                repel = repel * repel * uPointerStrength * (0.32 + grain * 0.68);
                vec2 screenDirection = normalize(metricDelta + vec2(0.0001));
                vec2 tangent = vec2(-screenDirection.y, screenDirection.x);
                vec2 irregularDirection = normalize(screenDirection + tangent * (aSeed - 0.5) * 0.95);
                vec2 ndcDirection = vec2(irregularDirection.x / max(0.1, uAspect), irregularDirection.y);

                float rippleEnvelope = (1.0 - smoothstep(0.045, 0.20, pointerDistance)) * uPointerStrength;
                float ripple = sin(pointerDistance * 118.0 - uTime * 5.0 + aSeed * 5.4) * rippleEnvelope * 0.0024;
                clipPosition.xy += ndcDirection * (repel * 0.045 + ripple) * clipPosition.w;

                gl_PointSize = clamp(aSize * uPixelRatio * (8.2 / max(1.0, -mvPosition.z)), 1.15, 8.2);
                gl_Position = clipPosition;
                vColor = aColor;
                vTwinkle = 0.78 + 0.22 * sin(uTime * 1.1 + aSeed * 31.0);
                vInteraction = repel + abs(ripple) * 5.0;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            varying float vTwinkle;
            varying float vInteraction;
            void main() {
                float d = distance(gl_PointCoord, vec2(0.5));
                if (d > 0.5) discard;
                float softGlow = smoothstep(0.5, 0.02, d);
                float core = smoothstep(0.17, 0.0, d);
                float alpha = softGlow * 0.68 + core;
                vec3 lightColor = vColor + vec3(0.36, 0.24, 0.07) * core;
                lightColor += vec3(0.18, 0.12, 0.025) * min(1.0, vInteraction);
                gl_FragColor = vec4(lightColor, alpha * vTwinkle * 0.96);
            }
        `,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });
    animatedMaterials.push(material);
    return material;
}

function asFloat32(value) {
    if (value instanceof Float32Array) return value;
    if (value instanceof ArrayBuffer) return new Float32Array(value);
    if (ArrayBuffer.isView(value)) return new Float32Array(value.buffer);
    return null;
}

function normalizeSurfaceData(record) {
    if (!record || Number(record.count) !== targetPointCount) return null;
    const data = {
        positions: asFloat32(record.positions),
        normals: asFloat32(record.normals),
        colors: asFloat32(record.colors),
        sizes: asFloat32(record.sizes),
        seeds: asFloat32(record.seeds),
        count: Number(record.count)
    };
    const count = data.count;
    if (!data.positions || data.positions.length !== count * 3 ||
        !data.normals || data.normals.length !== count * 3 ||
        !data.colors || data.colors.length !== count * 3 ||
        !data.sizes || data.sizes.length !== count ||
        !data.seeds || data.seeds.length !== count) return null;
    return data;
}

async function readPrecomputedParticleData() {
    const response = await fetch(particleDataUrl, {
        headers: { "Accept": "application/octet-stream" },
        cache: "force-cache"
    });
    if (!response.ok) {
        throw new Error(`Precomputed particle data returned ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength < 16) throw new Error("Precomputed particle data is truncated");
    const header = new DataView(buffer, 0, 16);
    const magic = String.fromCharCode(
        header.getUint8(0),
        header.getUint8(1),
        header.getUint8(2),
        header.getUint8(3)
    );
    const version = header.getUint32(4, true);
    const count = header.getUint32(8, true);
    const expectedBytes = 16 + count * 44;
    if (
        magic !== "XZP1"
        || version !== 1
        || count !== targetPointCount
        || buffer.byteLength !== expectedBytes
    ) {
        throw new Error("Precomputed particle data has an incompatible format");
    }
    let offset = 16;
    const take = (length) => {
        const value = new Float32Array(buffer, offset, length);
        offset += length * Float32Array.BYTES_PER_ELEMENT;
        return value;
    };
    return {
        positions: take(count * 3),
        normals: take(count * 3),
        colors: take(count * 3),
        sizes: take(count),
        seeds: take(count),
        count
    };
}

function mountParticleData(data) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
    geometry.setAttribute("aNormal", new THREE.BufferAttribute(data.normals, 3));
    geometry.setAttribute("aColor", new THREE.BufferAttribute(data.colors, 3));
    geometry.setAttribute("aSize", new THREE.BufferAttribute(data.sizes, 1));
    geometry.setAttribute("aSeed", new THREE.BufferAttribute(data.seeds, 1));
    const visiblePointCount = Math.min(data.count, INITIAL_PARTICLE_COUNT);
    geometry.setDrawRange(0, visiblePointCount);
    geometry.computeBoundingSphere();
    particleGeometry = geometry;
    currentDrawCount = visiblePointCount;
    maximumAdaptiveDrawCount = visiblePointCount;
    host.dataset.currentParticleCount = String(currentDrawCount);

    const core = new THREE.Points(geometry, createParticleMaterial());
    core.renderOrder = 1;
    particleGroup.add(core);
    if (particleCountLabel) particleCountLabel.textContent = visiblePointCount.toLocaleString("zh-CN");
    scheduleHardwareParticleBudget();
}

function openParticleCache() {
    return new Promise((resolve, reject) => {
        if (!("indexedDB" in window)) {
            reject(new Error("IndexedDB is unavailable"));
            return;
        }
        const request = indexedDB.open("xiezhi-particle-cache", 1);
        request.onupgradeneeded = () => {
            const database = request.result;
            if (!database.objectStoreNames.contains("surfaces")) database.createObjectStore("surfaces");
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function readParticleCache() {
    const database = await openParticleCache();
    try {
        return await new Promise((resolve, reject) => {
            const transaction = database.transaction("surfaces", "readonly");
            const request = transaction.objectStore("surfaces").get(cacheKey);
            request.onsuccess = () => resolve(normalizeSurfaceData(request.result));
            request.onerror = () => reject(request.error);
        });
    } finally {
        database.close();
    }
}

async function writeParticleCache(data) {
    const database = await openParticleCache();
    try {
        await new Promise((resolve, reject) => {
            const transaction = database.transaction("surfaces", "readwrite");
            transaction.objectStore("surfaces").put({ ...data, cachedAt: Date.now() }, cacheKey);
            transaction.oncomplete = resolve;
            transaction.onerror = () => reject(transaction.error);
        });
    } finally {
        database.close();
    }
}

function resize() {
    if (destroyed) return;
    const width = Math.max(1, host.clientWidth || 640);
    const height = Math.max(1, host.clientHeight || 640);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    pointerBounds = null;
    const pixelRatio = renderer.getPixelRatio();
    animatedMaterials.forEach((material) => {
        if (material.uniforms.uPixelRatio) material.uniforms.uPixelRatio.value = pixelRatio;
        if (material.uniforms.uAspect) material.uniforms.uAspect.value = width / height;
    });
}

function renderFrame(now) {
    if (!renderRunning || destroyed) return;
    animationFrame = requestAnimationFrame(renderFrame);
    const frameDelta = now - lastFrameTime;
    const isInteracting = pointerPresence > 0.025 || pointerPresenceTarget > 0;
    const frameInterval = isInteracting ? interactiveFrameInterval : idleFrameInterval;
    if (frameDelta < frameInterval * 0.82) return;
    lastFrameTime = now;

    if (isInteracting && particleGeometry && !reduceMotion) {
        if (frameDelta > interactiveFrameInterval * 1.55) slowInteractiveFrames += 1;
        else slowInteractiveFrames = Math.max(0, slowInteractiveFrames - 1);
        if (frameDelta <= interactiveFrameInterval * 1.18) fastInteractiveFrames += 1;
        else fastInteractiveFrames = Math.max(0, fastInteractiveFrames - 2);

        if (slowInteractiveFrames >= 6 && currentDrawCount > MIN_PARTICLE_COUNT) {
            updateVisibleParticleCount(Math.max(MIN_PARTICLE_COUNT, Math.floor(currentDrawCount * 0.78 / 500) * 500));
            slowInteractiveFrames = 0;
            fastInteractiveFrames = 0;
        } else if (fastInteractiveFrames >= 180 && currentDrawCount < maximumAdaptiveDrawCount) {
            updateVisibleParticleCount(Math.min(maximumAdaptiveDrawCount, currentDrawCount + 1000));
            fastInteractiveFrames = 0;
        }
    }

    const elapsed = clock.getElapsedTime();
    const pointerEase = reduceMotion ? 1 : 0.13;
    pointer.lerp(pointerTarget, pointerEase);
    pointerPresence += (pointerPresenceTarget - pointerPresence) * (reduceMotion ? 1 : 0.11);

    if (!reduceMotion) {
        root.rotation.y += (baseYaw + pointer.x * 0.065 - root.rotation.y) * 0.025;
        root.rotation.x += (pointer.y * 0.025 - root.rotation.x) * 0.025;
        root.position.copy(baseRootPosition);
        root.position.y += Math.sin(elapsed * 0.42) * 0.025;
        root.position.x += Math.cos(elapsed * 0.19) * 0.008;
        fieldGroup.rotation.y = elapsed * 0.012;
        particleGroup.rotation.y = Math.sin(elapsed * 0.12) * 0.012;
    }

    animatedMaterials.forEach((material) => {
        if (material.uniforms.uTime) material.uniforms.uTime.value = elapsed;
        if (material.uniforms.uPointer) material.uniforms.uPointer.value.copy(pointer);
        if (material.uniforms.uPointerStrength) material.uniforms.uPointerStrength.value = pointerPresence;
    });
    renderer.render(scene, camera);
}

function syncRenderState() {
    const shouldRender = isIntersecting && pageVisible && !destroyed;
    if (shouldRender && !renderRunning) {
        renderRunning = true;
        lastFrameTime = 0;
        animationFrame = requestAnimationFrame(renderFrame);
    } else if (!shouldRender && renderRunning) {
        renderRunning = false;
        cancelAnimationFrame(animationFrame);
    }
}

function flushPointerUpdate() {
    pointerUpdateFrame = 0;
    if (!pendingPointerPosition || destroyed) return;
    const position = pendingPointerPosition;
    pendingPointerPosition = null;
    const bounds = pointerBounds || host.getBoundingClientRect();
    pointerBounds = bounds;
    const localX = THREE.MathUtils.clamp(position.clientX - bounds.left, 0, bounds.width);
    const localY = THREE.MathUtils.clamp(position.clientY - bounds.top, 0, bounds.height);
    const xRatio = localX / Math.max(1, bounds.width);
    const yRatio = localY / Math.max(1, bounds.height);
    pointerTarget.set((xRatio - 0.5) * 2, -(yRatio - 0.5) * 2);
    if (pointerRipple) {
        pointerRipple.style.transform = `translate3d(${localX}px, ${localY}px, 0)`;
        pointerRipple.classList.add("is-active");
    }
}

function queuePointerUpdate(event) {
    pendingPointerPosition = { clientX: event.clientX, clientY: event.clientY };
    pointerPresenceTarget = 1;
    if (!pointerUpdateFrame) pointerUpdateFrame = requestAnimationFrame(flushPointerUpdate);
}

host.addEventListener("pointerenter", (event) => {
    pointerBounds = host.getBoundingClientRect();
    queuePointerUpdate(event);
});
host.addEventListener("pointermove", queuePointerUpdate, { passive: true });
host.addEventListener("pointerleave", () => {
    pendingPointerPosition = null;
    if (pointerUpdateFrame) cancelAnimationFrame(pointerUpdateFrame);
    pointerUpdateFrame = 0;
    pointerTarget.set(0, 0);
    pointerPresenceTarget = 0;
    if (pointerRipple) pointerRipple.classList.remove("is-active");
});
window.addEventListener("scroll", () => { pointerBounds = null; }, { passive: true });

const resizeObserver = new ResizeObserver(resize);
resizeObserver.observe(host);
const visibilityObserver = new IntersectionObserver((entries) => {
    isIntersecting = entries[0]?.isIntersecting ?? true;
    syncRenderState();
}, { threshold: 0.02 });
visibilityObserver.observe(host);

document.addEventListener("visibilitychange", () => {
    pageVisible = !document.hidden;
    syncRenderState();
});

window.addEventListener("pageshow", () => {
    pageVisible = !document.hidden;
    resize();
    syncRenderState();
});

createField();
createRings();
root.position.copy(baseRootPosition);
root.rotation.y = baseYaw;
resize();
syncRenderState();

async function initializeParticles() {
    setLoadingMessage("读取本地粒子缓存");
    try {
        const cached = await readParticleCache();
        if (cached && !destroyed) {
            mountParticleData(cached);
            resize();
            markReady("cache");
            return;
        }
    } catch (cacheError) {
        console.info("Particle cache is unavailable; loading the model directly", cacheError);
    }

    if (particleDataUrl) {
        try {
            setLoadingMessage("读取本地粒子数据");
            const data = await readPrecomputedParticleData();
            if (destroyed) return;
            mountParticleData(data);
            resize();
            markReady("precomputed");
            writeParticleCache(data).catch((cacheError) => {
                console.info("Unable to persist particle cache", cacheError);
            });
            return;
        } catch (particleDataError) {
            console.info(
                "Precomputed particle data is unavailable; loading the model directly",
                particleDataError
            );
        }
    }

    setLoadingMessage("准备载入模型");
    new GLTFLoader().load(
        modelUrl,
        async (gltf) => {
            try {
                setLoadingMessage("正在生成表面粒子");
                const data = await sampleModelSurface(gltf.scene);
                if (destroyed) return;
                mountParticleData(data);
                resize();
                markReady("model");
                writeParticleCache(data).catch((cacheError) => {
                    console.info("Unable to persist particle cache", cacheError);
                });
            } catch (samplingError) {
                if (destroyed) return;
                console.error("Unable to sample xiezhi.glb surface", samplingError);
                showError("獬豸模型表面解析失败，检测功能不受影响");
            }
        },
        updateLoadProgress,
        (loadError) => {
            if (destroyed) return;
            console.error("Unable to load xiezhi.glb", loadError);
            showError();
        }
    );
}

initializeParticles();

window.addEventListener("pagehide", (event) => {
    pageVisible = false;
    syncRenderState();
    if (event.persisted) return;
    destroyed = true;
    window.clearTimeout(startupBudgetTimer);
    if (pointerUpdateFrame) cancelAnimationFrame(pointerUpdateFrame);
    resizeObserver.disconnect();
    visibilityObserver.disconnect();
    scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) object.material.dispose();
    });
    renderer.dispose();
});
