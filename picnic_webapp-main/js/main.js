import { auth } from './firebase-config.js';
import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/12.0.0/firebase-auth.js";

const authBtn = document.getElementById('authBtn');

function setupLogoutButton() {
  authBtn.textContent = 'Logout';
  authBtn.classList.remove('login-button');
  authBtn.classList.add('logout-button');
  authBtn.href = '#';

  authBtn.addEventListener('click', (e) => {
    e.preventDefault();
    signOut(auth)
      .then(() => {
        alert('로그아웃 되었습니다.');
        window.location.href = 'index.html';
      })
      .catch((error) => {
        alert('로그아웃 중 오류: ' + error.message);
      });
  });
}

function setupLoginButton() {
  authBtn.textContent = 'Login';
  authBtn.classList.remove('logout-button');
  authBtn.classList.add('login-button');
  authBtn.href = 'index.html';

  authBtn.replaceWith(authBtn.cloneNode(true));
}

onAuthStateChanged(auth, (user) => {
  if (user) {
    console.log("로그인 상태 유지 중:", user.email);
    setupLogoutButton();
  } else {
    console.log("로그아웃 상태");
    setupLoginButton();
  }
});
