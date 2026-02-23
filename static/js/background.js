/**
 * background.js – Three.js animated particle network background
 * ──────────────────────────────────────────────────────────────
 * Creates a floating particle field with connecting lines.
 * Imported by every page for a cohesive visual identity.
 */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export function initBackground(canvasId = 'three-canvas') {
    const container = document.getElementById(canvasId);
    if (!container) return;

    // ── Scene setup ────────────────────────────────────────────────
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.z = 50;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // ── Particles ──────────────────────────────────────────────────
    const PARTICLE_COUNT = 200;
    const FIELD_SIZE = 80;

    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    const accentColor = new THREE.Color(0x6c63ff);
    const altColor = new THREE.Color(0x00d4aa);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;
        positions[i3]     = (Math.random() - 0.5) * FIELD_SIZE;
        positions[i3 + 1] = (Math.random() - 0.5) * FIELD_SIZE;
        positions[i3 + 2] = (Math.random() - 0.5) * FIELD_SIZE * 0.5;

        velocities[i3]     = (Math.random() - 0.5) * 0.02;
        velocities[i3 + 1] = (Math.random() - 0.5) * 0.02;
        velocities[i3 + 2] = (Math.random() - 0.5) * 0.01;

        const mixRatio = Math.random();
        const c = accentColor.clone().lerp(altColor, mixRatio);
        colors[i3]     = c.r;
        colors[i3 + 1] = c.g;
        colors[i3 + 2] = c.b;
    }

    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
        size: 0.4,
        vertexColors: true,
        transparent: true,
        opacity: 0.7,
        sizeAttenuation: true,
        blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    // ── Connection lines ───────────────────────────────────────────
    const LINE_DISTANCE = 12;
    const MAX_LINES = 600;
    const linePositions = new Float32Array(MAX_LINES * 6);
    const lineColors = new Float32Array(MAX_LINES * 6);

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

    const lineMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.15,
        blending: THREE.AdditiveBlending,
    });

    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    // ── Central glow sphere ────────────────────────────────────────
    const glowGeometry = new THREE.SphereGeometry(2, 32, 32);
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0x6c63ff,
        transparent: true,
        opacity: 0.08,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    scene.add(glow);

    // ── Mouse interaction ──────────────────────────────────────────
    const mouse = { x: 0, y: 0 };
    document.addEventListener('mousemove', (e) => {
        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    // ── Animation loop ─────────────────────────────────────────────
    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);
        const t = clock.getElapsedTime();

        // Update particle positions
        const pos = particleGeometry.attributes.position.array;
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const i3 = i * 3;
            pos[i3]     += velocities[i3];
            pos[i3 + 1] += velocities[i3 + 1];
            pos[i3 + 2] += velocities[i3 + 2];

            // Boundary wrap
            for (let j = 0; j < 3; j++) {
                const limit = j === 2 ? FIELD_SIZE * 0.25 : FIELD_SIZE * 0.5;
                if (pos[i3 + j] > limit)  pos[i3 + j] = -limit;
                if (pos[i3 + j] < -limit) pos[i3 + j] = limit;
            }
        }
        particleGeometry.attributes.position.needsUpdate = true;

        // Update lines between nearby particles
        let lineIdx = 0;
        const lp = lineGeometry.attributes.position.array;
        const lc = lineGeometry.attributes.color.array;
        for (let i = 0; i < PARTICLE_COUNT && lineIdx < MAX_LINES; i++) {
            for (let j = i + 1; j < PARTICLE_COUNT && lineIdx < MAX_LINES; j++) {
                const i3 = i * 3;
                const j3 = j * 3;
                const dx = pos[i3] - pos[j3];
                const dy = pos[i3+1] - pos[j3+1];
                const dz = pos[i3+2] - pos[j3+2];
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

                if (dist < LINE_DISTANCE) {
                    const li = lineIdx * 6;
                    lp[li]   = pos[i3];   lp[li+1] = pos[i3+1]; lp[li+2] = pos[i3+2];
                    lp[li+3] = pos[j3];   lp[li+4] = pos[j3+1]; lp[li+5] = pos[j3+2];

                    const alpha = 1 - dist / LINE_DISTANCE;
                    lc[li] = 0.42 * alpha; lc[li+1] = 0.39 * alpha; lc[li+2] = 1.0 * alpha;
                    lc[li+3] = 0.0 * alpha; lc[li+4] = 0.83 * alpha; lc[li+5] = 0.67 * alpha;
                    lineIdx++;
                }
            }
        }
        lineGeometry.setDrawRange(0, lineIdx * 2);
        lineGeometry.attributes.position.needsUpdate = true;
        lineGeometry.attributes.color.needsUpdate = true;

        // Gentle camera movement following mouse
        camera.position.x += (mouse.x * 5 - camera.position.x) * 0.02;
        camera.position.y += (mouse.y * 3 - camera.position.y) * 0.02;
        camera.lookAt(scene.position);

        // Pulsing glow
        glow.scale.setScalar(1 + Math.sin(t * 0.8) * 0.15);
        glow.material.opacity = 0.06 + Math.sin(t * 1.2) * 0.03;

        // Slow rotate particles
        particles.rotation.y = t * 0.03;
        particles.rotation.x = Math.sin(t * 0.02) * 0.1;

        renderer.render(scene, camera);
    }

    animate();

    // ── Resize handling ────────────────────────────────────────────
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}
