let isStartingTimestamp = true;
const video = document.getElementById("video");
const slider = document.getElementById("time-slider");
const timestampContainer = document.getElementById("timestamps-container");
const infoContainer = document.getElementById("timestamp-info-container");
const videoLabel = document.getElementById("video-label");
const speedSlider = document.getElementById("speed-slider");
const speedValue = document.getElementById("speed-value");
let hoveredMarker = null;
let selectedMarker = null;
const HOVER_THRESHOLD_PX = 20;
const STEP_MS = 1; // e.g. 50ms per step

speedSlider.addEventListener("input", (e) => {
  const exponent = parseFloat(speedSlider.value);
  const speed = Math.pow(2, exponent);
  setSpeed(speed);
});

function setSpeed(speed) {
  video.playbackRate = speed;
  speedSlider.value = Math.log2(speed);
  speedValue.textContent = `${roundFloat(speed, 2)}x`;
}

function seekVideo() {
  video.currentTime = slider.value;
}

function matchInterval(t, intervals) {
  let match = null;
  for (const interval of intervals) {
    const start = (parseFloat(interval.style.left) * videoDurationMs) / 100;
    const end =
      start + (parseFloat(interval.style.width) * videoDurationMs) / 100;
    if (t >= start && t <= end) {
      match = {
        state: interval.className.split(" ")[1],
        start: start,
        end: end,
      };
      return match;
    }
  }
}

function addTimestamp(state, time) {
  let percent = (time / video.duration) * 100;
  const marker = document.createElement("div");
  marker.dataset.type = isStartingTimestamp ? "start" : "end";
  marker.dataset.state = state;
  marker.className = `timestamp-marker ${marker.dataset.type} ${state}`;
  marker.style.left = `${percent}%`;
  marker.addEventListener("mousedown", dragMarker);
  const intervals = document.querySelectorAll(".timestamp-interval");
  const match = matchInterval(time * 30, intervals);

  if (match != null) return;
  if (marker.dataset.type === "end") {
    const markers = document.querySelectorAll(".timestamp-marker");
    const lastMarker = markers[markers.length - 1];
    let lastPercent = parseFloat(lastMarker.style.left);

    if (percent < lastPercent) {
      timestampContainer.insertBefore(marker, lastMarker);
      [marker.dataset.type, lastMarker.dataset.type] = [
        lastMarker.dataset.type,
        marker.dataset.type,
      ];
      [marker.dataset.state, lastMarker.dataset.state] = [
        lastMarker.dataset.state,
        marker.dataset.state,
      ];
      marker.className = `timestamp-marker ${marker.dataset.type} ${marker.dataset.state}`;
      lastMarker.className = `timestamp-marker ${lastMarker.dataset.type} ${lastMarker.dataset.state}`;
      [percent, lastPercent] = [lastPercent, percent];
    } else {
      timestampContainer.appendChild(marker);
    }
    displayInterval(lastPercent, percent, lastMarker.dataset.state);
  } else {
    timestampContainer.appendChild(marker);
  }
  isStartingTimestamp = !isStartingTimestamp;
  // add small delay
  setTimeout(() => {
    selectedMarker?.classList.remove("selected");
    selectedMarker = marker;
    selectedMarker.classList.add("selected");
  }, 100);
}

function addMarkersFromAnnotations() {
  videoTimestamps.forEach((ts) => {
    addTimestamp(ts.state, convertMsToS(ts.start));
    addTimestamp(ts.state, convertMsToS(ts.end));
  });
}

function dragMarker(event) {
  event.preventDefault();
  document.body.style.cursor = "grabbing";
  slider.style.cursor = "grabbing";
  let marker = event.currentTarget;
  const sliderRect = slider.getBoundingClientRect();

  if (!marker.classList || !marker.classList.contains("timestamp-marker")) {
    marker = hoveredMarker;
    if (!marker) {
      document.body.style.cursor = "default";
      slider.style.cursor = "default";
      return;
    }
  }

  function onMouseMove(e) {
    let x = e.clientX;
    let percent = ((x - sliderRect.left) / sliderRect.width) * 100;
    percent = clipMarkerPercent(marker, percent);
    marker.style.left = `${percent}%`;
    video.currentTime = (video.duration * percent) / 100;
    updateSingleInterval(marker);
  }
  function onMouseUp() {
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);
    document.body.style.cursor = "default";
    slider.style.cursor = "default";
  }
  document.addEventListener("mousemove", onMouseMove);
  document.addEventListener("mouseup", onMouseUp);
}

function updateSingleInterval(marker) {
  const markers = document.querySelectorAll(".timestamp-marker");
  const markerIndex = Array.from(markers).indexOf(marker);
  const intervalIndex = Math.floor(markerIndex / 2);
  const startMarker = markers[2 * intervalIndex];
  const endMarker = markers[2 * intervalIndex + 1];

  if (!endMarker) return;

  const startPercent = parseFloat(startMarker.style.left);
  const endPercent = parseFloat(endMarker.style.left);
  const widthPercent = endPercent - startPercent;
  const intervals = document.querySelectorAll(".timestamp-interval");
  const interval = intervals[intervalIndex];
  interval.style.left = `${startPercent}%`;
  interval.style.width = `${widthPercent}%`;
  const infos = document.querySelectorAll(".timestamp-info");
  const info = infos[intervalIndex];
  const start = Math.floor((startPercent / 100) * videoDurationMs);
  const end = Math.floor((endPercent / 100) * videoDurationMs);
  const startSpan = info.querySelector(".info-start");
  const endSpan = info.querySelector(".info-end");
  startSpan.textContent = `Start: ${start} ms`;
  endSpan.textContent = `End: ${end} ms`;
}

function displayInterval(startPercent, endPercent, state) {
  const width = endPercent - startPercent;
  let start = Math.floor((startPercent / 100) * videoDurationMs);
  let end = Math.floor((endPercent / 100) * videoDurationMs);
  const label = state === "stationnary" ? "Stationary" : "Seizure";
  const labelColor = state === "stationnary" ? "#05ce5a" : "dodgerblue";
  timestampContainer.insertAdjacentHTML(
    "beforeend",
    `<div class="timestamp-interval ${state}" style="left: ${startPercent}%; width: ${width}%;"></div>`,
  );
  infoContainer.innerHTML += `<div class="timestamp-info">
       <span class="info-state" style="color: ${labelColor}">${label}</span>
       <span class="info-start">Start: ${start} ms</span>
       <span class="info-end">End: ${end} ms</span>
       <button class="delete-btn">Delete</button>
     </div>`;
}

function removeTimestamp() {
  const markers = document.querySelectorAll(".timestamp-marker");
  if (markers.length === 0) return;
  if (markers[markers.length - 1].dataset.type === "end") {
    markers[markers.length - 2].remove();
    const intervals = document.querySelectorAll(".timestamp-interval");
    intervals[intervals.length - 1].remove();
    const infoContainer = document.querySelector(".timestamp-info-container");
    const infoItems = infoContainer.querySelectorAll(".timestamp-info");
    infoItems[infoItems.length - 1].remove();
  }
  isStartingTimestamp = true;
  markers[markers.length - 1].remove();
}

function clearAllAnnotations() {
  document.querySelectorAll(".timestamp-marker").forEach((m) => m.remove());
  document.querySelectorAll(".timestamp-interval").forEach((i) => i.remove());
  document.querySelectorAll(".timestamp-info").forEach((info) => info.remove());
  isStartingTimestamp = true;
}

function getAllTimestamps() {
  const markers = document.querySelectorAll(".timestamp-marker");
  let allTimestamps = [];
  for (let i = 0; i < markers.length; i += 2) {
    const startMarker = markers[i];
    const endMarker = markers[i + 1];

    if (endMarker) {
      const start = Math.floor(
        (parseFloat(startMarker.style.left) / 100) * videoDurationMs,
      );
      const end = Math.floor(
        (parseFloat(endMarker.style.left) / 100) * videoDurationMs,
      );
      const state = startMarker.dataset.state;

      allTimestamps.push({ state, start, end });
    }
  }
  return allTimestamps;
}

function saveTimestamps(allTimestamps) {
  const payload = {
    experiment,
    condition,
    video_number: videoNumber,
    timestamps: allTimestamps,
    user_comment: document.getElementById("user-input").value,
  };
  // if in review mode, send to /save_review, else to /save_timestamps
  const url = reviewMode ? saveReviewUrl : "/save_timestamps";
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.status !== "success") {
        alert(
          reviewMode
            ? "The review could not be saved."
            : "The annotations could not be saved.",
        );
      }
    })
    .catch((err) => console.error("Save error:", err));
}

function clipMarkerPercent(marker, percent) {
  const markers = Array.from(document.querySelectorAll(".timestamp-marker"));
  const sortedMarkers = markers
    .slice()
    .sort((a, b) => parseFloat(a.style.left) - parseFloat(b.style.left));
  const sortedIdx = sortedMarkers.indexOf(marker);
  const lower = sortedMarkers[sortedIdx - 1] || null;
  const upper = sortedMarkers[sortedIdx + 1] || null;
  const minPercent = lower ? parseFloat(lower.style.left) : 0;
  const maxPercent = upper ? parseFloat(upper.style.left) : 100;
  percent = Math.max(minPercent, Math.min(maxPercent * 0.999, percent)); // don't ask :3
  return percent;
}

function arrowPress(stepSize) {
  let videoTime = video.currentTime + (STEP_MS * stepSize) / 30;
  video.currentTime = Math.min(video.duration, Math.max(0, videoTime));
  if (selectedMarker) {
    let percent = (video.currentTime / video.duration) * 100;
    percent = clipMarkerPercent(selectedMarker, percent);
    selectedMarker.style.left = `${percent}%`;
    video.currentTime = (video.duration * percent) / 100;
    updateSingleInterval(selectedMarker);
  }
}

function handleKeyboardShortcuts(event) {
  const focused = document.activeElement;
  const stepPercent = (STEP_MS / videoDurationMs) * 100;
  const multStep = event.shiftKey ? 10 : 1; // Times 10 if shift is pressed
  if (
    focused.tagName === "TEXTAREA" ||
    (focused.tagName === "INPUT" && focused.type === "text")
  )
    return;
  switch (event.key.toLowerCase()) {
    case " ":
      event.preventDefault();
      video.paused ? video.play() : video.pause();
      break;
    case "s":
      event.preventDefault();
      addTimestamp("stationnary", video.currentTime);
      break;
    case "q":
      event.preventDefault();
      addTimestamp("cbm", video.currentTime);
      break;
    case "d":
      event.preventDefault();
      removeTimestamp();
      break;
    case "b":
      event.preventDefault();
      allTimestamps = getAllTimestamps();
      saveTimestamps(allTimestamps);
      break;
    case "n":
      event.preventDefault();
      allTimestamps = getAllTimestamps();
      saveTimestamps(allTimestamps);
      window.location.href = reviewMode ? reviewNextUrl : loadUnannotatedUrl;
      break;
    case "p":
      event.preventDefault();
      saveTimestamps(null);
      window.location.href = reviewMode ? reviewNextUrl : loadUnannotatedUrl;
      break;
    case "arrowright":
      event.preventDefault();
      arrowPress(multStep);
      break;
    case "arrowleft":
      event.preventDefault();
      arrowPress(-multStep);
      break;
    case "arrowup":
      event.preventDefault();
      selectedMarker.classList.remove("selected");
      selectedMarker = null;
      break;
    case "a":
      event.preventDefault();
      video.currentTime = 0;
      break;
    case "e":
      event.preventDefault();
      video.currentTime = video.duration;
      break;
    case "r":
      fetch(`/predict/${experiment}/${condition}/${videoNumber}`)
        .then((res) => res.json())
        .then((modelTimestamps) => {
          // 1) clear
          clearAllAnnotations();
          // 2) draw each interval from the model
          modelTimestamps.forEach((interval) => {
            // convert start/end (ms or frame count) to seconds
            const startSec = convertMsToS(interval.start);
            const endSec = convertMsToS(interval.end);
            // draw a pair of markers + interval
            addTimestamp(interval.state, startSec);
            addTimestamp(interval.state, endSec);
          });
        })
        .catch(console.error);
      break;
  }
  slider.value = video.currentTime;
  updateOverlay();
}

window.onload = function () {
  document.addEventListener("keydown", handleKeyboardShortcuts);
  video.addEventListener("loadedmetadata", () => {});
  document.getElementById("user-input").value = videoComment;

  video.addEventListener("play", () => {
    function rafUpdate() {
      slider.value = video.currentTime;
      if (!video.paused && !video.ended) {
        requestAnimationFrame(rafUpdate);
      }
    }
    requestAnimationFrame(rafUpdate);
  });
  video.currentTime = 0;
  slider.value = video.currentTime;
  slider.max = video.duration;
  setSpeed(1); // Default speed is 1x
  // Need to have the video duration in ms (time on the video) to get annotations in ms
  videoDurationMs = slider.max * 30; // Times 30 because the video is shot at 1000fps (1 frame = 1ms) then exported at 30fps
  addMarkersFromAnnotations();
};

function convertMsToS(time) {
  return time / 30;
}

slider.addEventListener("input", () => {
  seekVideo();
  updateOverlay();
});

let isPlaying = false;

video.addEventListener("play", () => {
  isPlaying = true;
  rafLoop();
});
video.addEventListener("pause", () => {
  isPlaying = false;
});

function rafLoop() {
  updateOverlay();
  if (isPlaying) requestAnimationFrame(rafLoop);
}

function updateOverlay() {
  const t = video.currentTime * 30;
  const intervals = document.querySelectorAll(".timestamp-interval");

  let match = matchInterval(t, intervals);

  if (!match) {
    video.style.filter = "";
    videoLabel.textContent = "Moving";
    videoLabel.style.color = "white";
  } else if (match.state === "stationnary") {
    video.style.filter =
      "sepia(1) saturate(.8) hue-rotate(70deg) brightness(0.8)";
    videoLabel.textContent = "Stationary";
    videoLabel.style.color = "#05ce5a";
  } else {
    video.style.filter =
      "sepia(1) saturate(.8) hue-rotate(180deg) brightness(0.8)";
    videoLabel.textContent = "Seizure";
    videoLabel.style.color = "dodgerblue";
  }
}

function roundFloat(number, precision) {
  return Math.round(number * Math.pow(10, precision)) / Math.pow(10, precision);
}

infoContainer.addEventListener("click", (e) => {
  if (!e.target.classList.contains("delete-btn")) return;

  const infoEl = e.target.closest(".timestamp-info");
  const infos = Array.from(infoContainer.querySelectorAll(".timestamp-info"));
  const idx = infos.indexOf(infoEl);
  if (idx < 0) return;

  const markers = document.querySelectorAll(".timestamp-marker");
  const startMarker = markers[2 * idx];
  const endMarker = markers[2 * idx + 1];
  if (endMarker) endMarker.remove();
  if (startMarker) startMarker.remove();

  const intervals = document.querySelectorAll(".timestamp-interval");
  if (intervals[idx]) intervals[idx].remove();
  infoEl.remove();
  updateOverlay();
  isStartingTimestamp = true;
});

document.body.addEventListener("mousemove", (e) => {
  const rect = timestampContainer.getBoundingClientRect();
  const x = e.clientX - rect.left;
  let closest = null,
    minDist = Infinity;

  timestampContainer.querySelectorAll(".timestamp-marker").forEach((marker) => {
    const markerX = (parseFloat(marker.style.left) / 100) * rect.width;
    const dist = Math.abs(markerX - x);
    if (dist < minDist) {
      minDist = dist;
      closest = marker;
    }
  });

  if (minDist < HOVER_THRESHOLD_PX) {
    if (document.body.style.cursor !== "grabbing") {
      slider.style.cursor = "grab";
      document.body.style.cursor = "grab";
    }
    if (hoveredMarker !== closest) {
      hoveredMarker?.classList.remove("hover");
      hoveredMarker = closest;
      hoveredMarker.classList.add("hover");
    }
  } else {
    slider.style.cursor = "default";
    document.body.style.cursor = "default";
    hoveredMarker?.classList.remove("hover");
    hoveredMarker = null;
  }
});

document.body.addEventListener("click", (e) => {
  if (!hoveredMarker) {
    if (selectedMarker) {
      selectedMarker.classList.remove("selected");
      selectedMarker = null;
    }
    return;
  }
  selectedMarker?.classList.remove("selected");
  selectedMarker = hoveredMarker;
  selectedMarker.classList.add("selected");
  // if clicking inside the associated interval, update the video time, else, ignore
  const intervals = document.querySelectorAll(".timestamp-interval");
  const match = matchInterval(video.currentTime * 30, intervals);
  if (match)
    slider.value = video.currentTime =
      (video.duration * parseFloat(selectedMarker.style.left)) / 100;
});

document.body.addEventListener("mousedown", (e) => {
  if (hoveredMarker) {
    dragMarker(e);
  }
});
