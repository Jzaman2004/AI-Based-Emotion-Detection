const emotionKeys = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "hate"];

const capturedImage = document.getElementById("captured-image");
const emptyState = document.getElementById("empty-state");
const retryBtn = document.getElementById("retry-btn");

function setDefaults() {
  document.getElementById("person-visible").textContent = "Capture image first";
  document.getElementById("gender").textContent = "Capture image first";
  document.getElementById("race").textContent = "Capture image first";
  document.getElementById("dominant-emotion").textContent = "Capture image first";

  emotionKeys.forEach((emotion) => {
    document.getElementById(`${emotion}-bar`).style.width = "0%";
    document.getElementById(`${emotion}-percent`).textContent = "0%";
  });
}

function updateResults(data) {
  document.getElementById("person-visible").textContent = data.person_visible ? "Yes" : "No";
  document.getElementById("gender").textContent = data.gender || "unknown";
  document.getElementById("race").textContent = data.race || "unknown";
  document.getElementById("dominant-emotion").textContent = data.dominant_emotion || "—";

  const emotions = data.emotions || {};
  emotionKeys.forEach((emotion) => {
    const value = Number.isFinite(Number(emotions[emotion])) ? Number(emotions[emotion]) : 0;
    const clamped = Math.max(0, Math.min(100, Math.round(value)));
    document.getElementById(`${emotion}-bar`).style.width = `${clamped}%`;
    document.getElementById(`${emotion}-percent`).textContent = `${clamped}%`;
  });
}

function loadPageData() {
  setDefaults();

  const imageData = sessionStorage.getItem("capturedImage");
  const analysisRaw = sessionStorage.getItem("analysisData");

  if (!imageData || !analysisRaw) {
    emptyState.hidden = false;
    return;
  }

  capturedImage.src = imageData;
  capturedImage.hidden = false;

  try {
    const analysis = JSON.parse(analysisRaw);
    updateResults(analysis);
    emptyState.hidden = true;
  } catch {
    emptyState.hidden = false;
  }
}

function onRetry() {
  sessionStorage.removeItem("capturedImage");
  sessionStorage.removeItem("analysisData");
  window.location.href = "/";
}

retryBtn.addEventListener("click", onRetry);
window.addEventListener("DOMContentLoaded", loadPageData);
