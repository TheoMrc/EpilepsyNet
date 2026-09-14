class Confetti {
  constructor() {
    this.confetti = [];
    this.init();
  }

  init() {
    this.createConfetti();
  }

  createConfetti() {
    const confettiContainer = document.getElementById("confettiContainer");
    const confettiCount = 100;

    for (let i = 0; i < confettiCount; i++) {
      const confetti = this.createSingleConfetti(i);
      confettiContainer.appendChild(confetti.element);
      this.confetti.push(confetti);
    }
  }

  createSingleConfetti(index) {
    const confettiElement = document.createElement("div");
    confettiElement.className = "confetti";

    const size = 5 + Math.random() * 10;
    confettiElement.style.width = `${size}px`;
    confettiElement.style.height = `${size}px`;

    const colors = [
      "#ff0000",
      "#00ff00",
      "#0000ff",
      "#ffff00",
      "#ff00ff",
      "#00ffff",
    ];
    confettiElement.style.backgroundColor =
      colors[Math.floor(Math.random() * colors.length)];

    const startX = -20 + Math.random() * 140;
    confettiElement.style.setProperty("--startX", `${startX}vw`);

    const driftX = (Math.random() - 0.5) * 40;
    confettiElement.style.setProperty("--driftX", `${driftX}vw`);

    const rotation = Math.random() * 720 - 360;
    confettiElement.style.setProperty("--rotation", `${rotation}deg`);

    const delay = Math.random() * 5;
    confettiElement.style.setProperty("--delay", `${delay}s`);

    const duration = 3 + Math.random() * 3;
    confettiElement.style.animationDuration = `${duration}s`;

    return {
      element: confettiElement,
    };
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const confetti = new Confetti();
});
