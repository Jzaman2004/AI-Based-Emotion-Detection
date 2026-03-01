const emotionKeys = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "hate"];
const feedbackStorageKey = "ethicalFeedbackCounts";

const capturedImage = document.getElementById("captured-image");
const emptyState = document.getElementById("empty-state");
const retryBtn = document.getElementById("retry-btn");
const feedbackOptions = document.getElementById("feedback-options");
const feedbackQuestionKeys = [
  "fairness-concern",
  "context-mismatch",
  "emotion-uncertain",
  "high-stakes-limit",
];
const feedbackAnswers = ["yes", "no"];

function toSafeCount(value) {
  return Number.isFinite(Number(value)) ? Math.max(0, Number(value)) : 0;
}

function normalizeFeedbackCounts(rawCounts) {
  const normalized = {};

  feedbackQuestionKeys.forEach((questionKey) => {
    const source = rawCounts && typeof rawCounts === "object" ? rawCounts[questionKey] : null;
    normalized[questionKey] = {
      yes: toSafeCount(source?.yes),
      no: toSafeCount(source?.no),
    };
  });

  return normalized;
}

function getFeedbackCounts() {
  const raw = sessionStorage.getItem(feedbackStorageKey);
  if (!raw) return normalizeFeedbackCounts({});

  try {
    const parsed = JSON.parse(raw);
    return normalizeFeedbackCounts(parsed);
  } catch {
    return normalizeFeedbackCounts({});
  }
}

function setFeedbackCounts(counts) {
  sessionStorage.setItem(feedbackStorageKey, JSON.stringify(counts));
}

function renderFeedbackCounts() {
  if (!feedbackOptions) return;

  const counts = getFeedbackCounts();
  const voteButtons = feedbackOptions.querySelectorAll(".feedback-vote[data-feedback-key][data-feedback-answer]");

  voteButtons.forEach((button) => {
    const key = button.getAttribute("data-feedback-key");
    const answer = button.getAttribute("data-feedback-answer");
    const countEl = button.querySelector(".feedback-count");
    if (!key || !answer || !countEl) return;

    const current = toSafeCount(counts[key]?.[answer]);
    countEl.textContent = String(current);
  });
}

function incrementFeedbackCount(feedbackKey, feedbackAnswer) {
  if (!feedbackKey || !feedbackAnswer || !feedbackAnswers.includes(feedbackAnswer)) return;

  const counts = getFeedbackCounts();
  const current = toSafeCount(counts[feedbackKey]?.[feedbackAnswer]);
  counts[feedbackKey][feedbackAnswer] = current + 1;
  setFeedbackCounts(counts);
  renderFeedbackCounts();
}

function onFeedbackClick(event) {
  const button = event.target.closest(".feedback-vote[data-feedback-key][data-feedback-answer]");
  if (!button) return;
  incrementFeedbackCount(button.getAttribute("data-feedback-key"), button.getAttribute("data-feedback-answer"));
}

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
  renderFeedbackCounts();

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
  window.location.replace("/");
}

retryBtn.addEventListener("click", onRetry);
if (feedbackOptions) {
  feedbackOptions.addEventListener("click", onFeedbackClick);
}
window.addEventListener("DOMContentLoaded", loadPageData);
