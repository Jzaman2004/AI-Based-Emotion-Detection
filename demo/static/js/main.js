const captureBtn = document.getElementById("capture-btn");
const video = document.getElementById("camera-feed");
const canvas = document.getElementById("capture-canvas");
const loading = document.getElementById("loading");
const statusMessage = document.getElementById("status-message");
const consentOverlay = document.getElementById("consent-overlay");
const consentAgreeBtn = document.getElementById("consent-agree-btn");

const captureCooldownMs = 5000;
let lastCaptureTime = 0;
let mediaStream = null;

function setStatus(message = "") {
  statusMessage.textContent = message;
}

function setLoading(isLoading) {
  if (loading) {
    loading.hidden = !isLoading;
  }
  captureBtn.disabled = isLoading;
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function showConsentPopup() {
  return new Promise((resolve) => {
    if (!consentOverlay || !consentAgreeBtn) {
      resolve();
      return;
    }

    consentOverlay.hidden = false;

    const onAgree = () => {
      consentOverlay.hidden = true;
      resolve();
    };

    consentAgreeBtn.addEventListener("click", onAgree, { once: true });
  });
}

async function startCamera() {
  stopCamera();
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = mediaStream;
    video.hidden = false;
    canvas.hidden = true;
    setStatus("");
    return true;
  } catch (error) {
    setStatus("Camera access denied or unavailable. Please allow camera permissions.");
    return false;
  }
}

function createResizedCanvas(maxWidth = 1024, maxHeight = 1024) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;

  if (!sourceWidth || !sourceHeight) {
    throw new Error("No camera frame available");
  }

  const scale = Math.min(maxWidth / sourceWidth, maxHeight / sourceHeight, 1);
  const targetWidth = Math.round(sourceWidth * scale);
  const targetHeight = Math.round(sourceHeight * scale);

  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, targetWidth, targetHeight);
  return canvas;
}

function canvasToDataUrl(imageCanvas, quality = 0.8) {
  return imageCanvas.toDataURL("image/jpeg", quality);
}

async function captureAndAnalyze() {
  const now = Date.now();
  const remaining = captureCooldownMs - (now - lastCaptureTime);
  if (remaining > 0) {
    setStatus(`Please wait ${Math.ceil(remaining / 1000)}s before capturing again.`);
    return;
  }

  try {
    setLoading(true);
    setStatus("");

    const resizedCanvas = createResizedCanvas(1024, 1024);
    const dataUrl = canvasToDataUrl(resizedCanvas, 0.8);

    const formData = new FormData();
    formData.append("image", dataUrl);

    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok || !payload.success) {
      throw new Error(payload.message || "API error while analyzing image");
    }

    sessionStorage.setItem("capturedImage", dataUrl);
    sessionStorage.setItem("analysisData", JSON.stringify(payload.data));
    lastCaptureTime = Date.now();
    stopCamera();
    window.location.href = "/results";
  } catch (error) {
    setStatus(error.message || "Unable to analyze image. Please try again.");
  } finally {
    setLoading(false);
  }
}

captureBtn.addEventListener("click", captureAndAnalyze);

window.addEventListener("DOMContentLoaded", async () => {
  setLoading(false);
  captureBtn.disabled = true;
  await showConsentPopup();
  const started = await startCamera();
  if (!started) {
    captureBtn.disabled = true;
    return;
  }
  captureBtn.disabled = false;
});
