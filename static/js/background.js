/**
 * background.js – Minimalist no-op.
 *
 * The previous Three.js gradient mesh has been replaced by a static CSS-only
 * background (see `body::before` / `body::after` in `style.css`). This export
 * remains for backwards compatibility with existing `initBackground()` imports.
 */
export function initBackground() {
    const canvas = document.getElementById('three-canvas');
    if (canvas) canvas.remove();
}
