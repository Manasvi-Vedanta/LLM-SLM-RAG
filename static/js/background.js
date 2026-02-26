/**
 * background.js – Three.js animated gradient mesh background
 * ───────────────────────────────────────────────────────────
 * Creates a smooth, organic undulating surface with subtle color
 * gradients that react gently to mouse movement. Professional and
 * seamless — no disconnected particles, no visual noise.
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export function initBackground(canvasId = 'three-canvas') {
    const container = document.getElementById(canvasId);
    if (!container) return;

    // ── Scene ──────────────────────────────────────────────────────
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        500
    );
    camera.position.set(0, 20, 50);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // ── Colour palette ─────────────────────────────────────────────
    const COL_DEEP    = new THREE.Color(0x0a0a1a);  // near-black blue
    const COL_ACCENT  = new THREE.Color(0x6c63ff);  // brand purple
    const COL_TEAL    = new THREE.Color(0x00d4aa);  // accent teal
    const COL_MID     = new THREE.Color(0x1a1a3e);  // dark indigo

    // ── Gradient plane geometry ────────────────────────────────────
    const SEGMENTS = 128;
    const SIZE     = 120;
    const geometry = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
    geometry.rotateX(-Math.PI * 0.45);

    // Per-vertex colour attribute
    const count  = geometry.attributes.position.count;
    const colors = new Float32Array(count * 3);

    const pos = geometry.attributes.position.array;
    for (let i = 0; i < count; i++) {
        const x = pos[i * 3];
        const z = pos[i * 3 + 2];
        const nx = (x / SIZE) + 0.5;
        const nz = (z / SIZE) + 0.5;

        const t   = nx * 0.6 + nz * 0.4;
        const col = COL_DEEP.clone().lerp(COL_MID, t);

        const cx = nx - 0.5, cz = nz - 0.5;
        const distFromCenter = Math.sqrt(cx * cx + cz * cz) * 2;
        const accentMix = Math.max(0, 1 - distFromCenter) * 0.25;
        col.lerp(COL_ACCENT, accentMix);

        colors[i * 3]     = col.r;
        colors[i * 3 + 1] = col.g;
        colors[i * 3 + 2] = col.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.55,
        wireframe: false,
        side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.y = -8;
    scene.add(mesh);

    // ── Soft floating orbs (ambient glow) ──────────────────────────
    const orbs = [];
    const orbData = [
        { color: 0x6c63ff, size: 3.0, x: -15, y:  4, z: -10, speed: 0.3 },
        { color: 0x00d4aa, size: 2.2, x:  20, y:  6, z:   5, speed: 0.4 },
        { color: 0x6c63ff, size: 1.8, x:   5, y:  2, z:  15, speed: 0.25 },
        { color: 0x00d4aa, size: 2.5, x: -25, y:  8, z:  10, speed: 0.35 },
        { color: 0x4a45c2, size: 2.0, x:  12, y:  3, z: -18, speed: 0.28 },
    ];

    for (const d of orbData) {
        const geo = new THREE.SphereGeometry(d.size, 24, 24);
        const mat = new THREE.MeshBasicMaterial({
            color: d.color,
            transparent: true,
            opacity: 0.06,
        });
        const orb = new THREE.Mesh(geo, mat);
        orb.position.set(d.x, d.y, d.z);
        orb.userData = { ...d, baseY: d.y, baseX: d.x };
        scene.add(orb);
        orbs.push(orb);
    }

    // ── Mouse tracking ─────────────────────────────────────────────
    const mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    document.addEventListener('mousemove', (e) => {
        mouse.targetX = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.targetY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    // ── Store original Y values for wave animation ─────────────────
    const originalY = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        originalY[i] = pos[i * 3 + 1];
    }

    // ── Animation loop ─────────────────────────────────────────────
    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);
        const t = clock.getElapsedTime();

        // Smooth mouse interpolation
        mouse.x += (mouse.targetX - mouse.x) * 0.03;
        mouse.y += (mouse.targetY - mouse.y) * 0.03;

        // ── Undulating wave on the mesh ────────────────────────────
        const posAttr = geometry.attributes.position;
        const colAttr = geometry.attributes.color;
        const arr     = posAttr.array;
        const carr    = colAttr.array;

        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            const x  = arr[ix];
            const z  = arr[ix + 2];

            // Layered sine waves for organic motion
            const wave1 = Math.sin(x * 0.06 + t * 0.4) * 1.8;
            const wave2 = Math.sin(z * 0.08 + t * 0.3) * 1.2;
            const wave3 = Math.sin((x + z) * 0.04 + t * 0.5) * 0.8;
            const mouseInfluence =
                Math.sin(x * 0.05 + mouse.x * 3 + t * 0.2) *
                Math.cos(z * 0.05 + mouse.y * 3 + t * 0.2) * 1.0;

            arr[ix + 1] = originalY[i] + wave1 + wave2 + wave3 + mouseInfluence;

            // Subtly shift colours with the wave height
            const height = (arr[ix + 1] + 5) / 10;
            const nx = (x / SIZE) + 0.5;
            const nz = (z / SIZE) + 0.5;
            const baseT = nx * 0.6 + nz * 0.4;
            const col = COL_DEEP.clone().lerp(COL_MID, baseT);

            const peakTint = Math.max(0, height) * 0.15;
            const tealOrPurple = Math.sin(t * 0.2 + nx * 3) > 0 ? COL_TEAL : COL_ACCENT;
            col.lerp(tealOrPurple, peakTint);

            carr[ix]     = col.r;
            carr[ix + 1] = col.g;
            carr[ix + 2] = col.b;
        }
        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;

        // ── Animate orbs ───────────────────────────────────────────
        for (const orb of orbs) {
            const d = orb.userData;
            orb.position.y = d.baseY + Math.sin(t * d.speed) * 3;
            orb.position.x = d.baseX + Math.sin(t * d.speed * 0.7 + 1) * 4;
            orb.material.opacity = 0.04 + Math.sin(t * d.speed * 1.2) * 0.025;
            const scale = 1 + Math.sin(t * d.speed * 0.5) * 0.15;
            orb.scale.setScalar(scale);
        }

        // ── Gentle camera sway ─────────────────────────────────────
        camera.position.x += (mouse.x * 6 - camera.position.x) * 0.015;
        camera.position.y += (20 + mouse.y * 4 - camera.position.y) * 0.015;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
    }

    animate();

    // ── Resize ─────────────────────────────────────────────────────
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}
