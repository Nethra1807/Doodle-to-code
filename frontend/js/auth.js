/* Auth logic (localStorage tokens) */

const API_URL = 'http://127.0.0.1:5000';

export function getToken() {
  return localStorage.getItem('authToken');
}

export function isLoggedIn() {
  return !!getToken();
}

export function logout() {
  localStorage.removeItem('authToken');
  localStorage.removeItem('authName');
  window.location.href = 'login.html';
}

export async function login(email, password) {
  try {
    const res = await fetch(`${API_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || 'Login failed');
    }
    localStorage.setItem('authToken', data.token);
    localStorage.setItem('authName', data.name);
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

export async function signup(name, email, password, confirm, phone) {
  try {
    const res = await fetch(`${API_URL}/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, confirm, phone })
    });
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || 'Signup failed');
    }
    localStorage.setItem('authToken', data.token);
    localStorage.setItem('authName', data.name);
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// Ensure the user is authenticated on the main app
export async function requireAuth() {
    if (!isLoggedIn()) {
        window.location.href = 'login.html';
        return;
    }
    // Optional: verify token internally
    try {
        const res = await fetch(`${API_URL}/verify-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: getToken() })
        });
        if (!res.ok) {
            logout();
        }
    } catch {
        // If network error, just let them try. prediction endpoint will also catch.
    }
}
