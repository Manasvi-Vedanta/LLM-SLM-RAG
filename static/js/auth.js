/**
 * auth.js – Shared authentication helpers
 * ────────────────────────────────────────
 * Token management, auth state, and API calls.
 */

const API_BASE = '';

// ── Token management ─────────────────────────────────────────────
export function getToken() {
    return localStorage.getItem('rag_token');
}

export function setToken(token) {
    localStorage.setItem('rag_token', token);
}

export function removeToken() {
    localStorage.removeItem('rag_token');
    localStorage.removeItem('rag_user');
}

export function getUser() {
    const raw = localStorage.getItem('rag_user');
    return raw ? JSON.parse(raw) : null;
}

export function setUser(user) {
    localStorage.setItem('rag_user', JSON.stringify(user));
}

// ── Auth API calls ───────────────────────────────────────────────
export async function signup(username, email, password) {
    const res = await fetch(`${API_BASE}/api/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Signup failed');
    setToken(data.token);
    setUser(data.user);
    return data;
}

export async function login(username, password) {
    const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Login failed');
    setToken(data.token);
    setUser(data.user);
    return data;
}

export function logout() {
    removeToken();
    window.location.href = '/';
}

// ── Auth-required fetch helper ──────────────────────────────────
export async function authFetch(url, options = {}) {
    const token = getToken();
    if (!token) {
        window.location.href = '/login';
        throw new Error('Not authenticated');
    }
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        ...(options.headers || {}),
    };
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
        removeToken();
        window.location.href = '/login';
        throw new Error('Session expired');
    }
    return res;
}

// ── Check if user is logged in ──────────────────────────────────
export function isAuthenticated() {
    return !!getToken();
}

// ── Require auth (redirect to login) ────────────────────────────
export function requireAuth() {
    if (!isAuthenticated()) {
        window.location.href = '/login';
        return false;
    }
    return true;
}
