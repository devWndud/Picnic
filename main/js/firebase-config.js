import { initializeApp } from "https://www.gstatic.com/firebasejs/12.0.0/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/12.0.0/firebase-analytics.js";
import {
  getAuth,
  setPersistence,
  browserLocalPersistence
} from "https://www.gstatic.com/firebasejs/12.0.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/12.0.0/firebase-firestore.js";

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
const analytics = getAnalytics(app);

const auth = getAuth(app);

setPersistence(auth, browserLocalPersistence)
  .catch((error) => {
    console.error("Auth persistence 설정 실패:", error);
  });

const db = getFirestore(app);

export { app, analytics, auth, db };
