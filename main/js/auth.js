import { auth, db } from './firebase-config.js';
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/12.0.0/firebase-auth.js";
import {
  doc, setDoc
} from "https://www.gstatic.com/firebasejs/12.0.0/firebase-firestore.js";


const validDeviceCodes = ["A7X2bQ9cLmY1"];

const signupForm = document.getElementById('signupForm');
if (signupForm) {
  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = signupForm.signupEmail.value.trim();
    const password = signupForm.signupPassword.value.trim();
    const deviceCode = signupForm.deviceCode.value.trim();

    if (!email || !password || password.length < 6 || !deviceCode) {
      alert('모든 항목을 올바르게 입력해주세요.');
      return;
    }

    if (!validDeviceCodes.includes(deviceCode)) {
      alert('유효하지 않은 기기 코드입니다.');
      return;
    }

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      await setDoc(doc(db, "users", user.uid), {
        email,
        deviceCode,
        createdAt: new Date()
      });

      alert('회원가입 성공! 기능 페이지로 이동합니다.');
      window.location.href = 'main.html';

    } catch (error) {
      alert('회원가입 실패: ' + error.message);
    }
  });
}

const loginForm = document.getElementById('loginForm');
if (loginForm) {
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = loginForm.loginEmail.value.trim();
    const password = loginForm.loginPassword.value.trim();

    try {
      await signInWithEmailAndPassword(auth, email, password);
      window.location.href = 'main.html';
    } catch (error) {
      alert('로그인 실패: ' + error.message);
    }
  });
}

export function logout() {
  signOut(auth)
    .then(() => {
      alert('로그아웃 되었습니다.');
      window.location.href = 'index.html';
    })
    .catch((error) => {
      alert('로그아웃 오류: ' + error.message);
    });
}

onAuthStateChanged(auth, (user) => {
  const loginBtn = document.getElementById("loginBtn") || document.getElementById("login-btn") || document.getElementById("authBtn");
  
  if (!loginBtn) return;

  if (user) {
    console.log("로그인 유지 중:", user.email);
    loginBtn.textContent = "로그아웃";
    loginBtn.onclick = (e) => {
      e?.preventDefault?.();
      logout();
    };
    loginBtn.classList.remove('login-button');
    loginBtn.classList.add('logout-button');
    loginBtn.href = '#';
  } else {
    console.log("로그아웃 상태");
    loginBtn.textContent = "로그인";
    loginBtn.onclick = () => window.location.href = "index.html";
    loginBtn.classList.remove('logout-button');
    loginBtn.classList.add('login-button');
    loginBtn.href = 'index.html';
  }
});

const checkbox = document.getElementById('agreeTerms');
    const modal = document.getElementById('termsModal');
    const closeBtn = document.querySelector('.modal .close');

    checkbox.addEventListener('change', (e) => {
      if (e.target.checked) {
        modal.style.display = 'block';
        checkbox.checked = true;
      }
    });

    closeBtn.onclick = function () {
      modal.style.display = 'none';
    };

    window.onclick = function (event) {
      if (event.target == modal) {
        modal.style.display = 'none';
      }
    };