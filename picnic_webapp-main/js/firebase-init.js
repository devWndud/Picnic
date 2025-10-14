import { initializeApp } from "https://www.gstatic.com/firebasejs/12.0.0/firebase-app.js";
import {
  getAuth,
  onAuthStateChanged,
  signOut
} from "https://www.gstatic.com/firebasejs/12.0.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyDiMA0QO-vD0SxoGqky0GZqiAplFnoqMmg",
    authDomain: "picnic-cd13b.firebaseapp.com",
    projectId: "picnic-cd13b",
    storageBucket: "picnic-cd13b.appspot.com",
    messagingSenderId: "420041964633",
    appId: "1:420041964633:web:c2da38e4e33dd94bb24548",
    measurementId: "G-FRMM7MGFWC"
  };

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

onAuthStateChanged(auth, (user) => {
  const loginBtn = document.getElementById("login-btn");
  if (user) {
    console.log("로그인 유지 중:", user.email);
    if (loginBtn) {
      loginBtn.textContent = "로그아웃";
      loginBtn.onclick = () => {
        signOut(auth).then(() => {
          console.log("로그아웃 완료");
          window.location.reload();
        });
      };
    }
  } else {
    console.log("로그인 안됨");
    if (loginBtn) {
      loginBtn.textContent = "로그인";
      loginBtn.onclick = () => {
        window.location.href = "/index.html";
      };
    }
  }
});
