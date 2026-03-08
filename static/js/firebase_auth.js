/**
 * CardioSense — Firebase Auth (minimal JS).
 * Handles sign-up and sign-in with Firebase Auth, then sends the
 * ID token to the Flask backend to create a server-side session.
 */

// Firebase config — injected from template
const firebaseConfig = window.FIREBASE_CONFIG || {};

// Initialize Firebase
import('https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js').then((mod) => {
    const app = mod.initializeApp(firebaseConfig);
    return import('https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js');
}).then((authMod) => {
    window.firebaseAuth = authMod;
    window.auth = authMod.getAuth();
}).catch(err => console.error('Firebase init error:', err));


/**
 * Sign in with email and password.
 */
async function csLogin(email, password) {
    const btn = document.getElementById('login-btn');
    const errDiv = document.getElementById('auth-error');
    if (btn) btn.disabled = true;
    if (errDiv) errDiv.textContent = '';

    try {
        const { signInWithEmailAndPassword } = window.firebaseAuth;
        const result = await signInWithEmailAndPassword(window.auth, email, password);
        const idToken = await result.user.getIdToken();

        const resp = await fetch('/auth/session-login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idToken }),
        });
        const data = await resp.json();

        if (data.success) {
            window.location.href = data.redirect;
        } else {
            if (errDiv) errDiv.textContent = data.error || 'Login failed.';
        }
    } catch (e) {
        let msg = 'Login failed.';
        if (e.code === 'auth/user-not-found') msg = 'No account found with this email.';
        else if (e.code === 'auth/wrong-password') msg = 'Incorrect password.';
        else if (e.code === 'auth/invalid-email') msg = 'Invalid email address.';
        else if (e.code === 'auth/invalid-credential') msg = 'Invalid email or password.';
        if (errDiv) errDiv.textContent = msg;
    } finally {
        if (btn) btn.disabled = false;
    }
}


/**
 * Sign up with email and password.
 */
async function csSignup(email, password, fullName, role) {
    const btn = document.getElementById('signup-btn');
    const errDiv = document.getElementById('auth-error');
    if (btn) btn.disabled = true;
    if (errDiv) errDiv.textContent = '';

    try {
        const { createUserWithEmailAndPassword } = window.firebaseAuth;
        const result = await createUserWithEmailAndPassword(window.auth, email, password);
        const idToken = await result.user.getIdToken();

        const resp = await fetch('/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idToken, role, fullName }),
        });
        const data = await resp.json();

        if (data.success) {
            window.location.href = data.redirect;
        } else {
            if (errDiv) errDiv.textContent = data.error || 'Signup failed.';
        }
    } catch (e) {
        let msg = 'Signup failed.';
        if (e.code === 'auth/email-already-in-use') msg = 'This email is already registered.';
        else if (e.code === 'auth/weak-password') msg = 'Password must be at least 6 characters.';
        else if (e.code === 'auth/invalid-email') msg = 'Invalid email address.';
        if (errDiv) errDiv.textContent = msg;
    } finally {
        if (btn) btn.disabled = false;
    }
}


// Expose functions globally
window.csLogin = csLogin;
window.csSignup = csSignup;
