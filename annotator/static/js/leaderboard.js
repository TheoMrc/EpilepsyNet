// shared pastel palette
const pastelPalette = [
  "#FFC4C4",
  "#FFE5A5",
  "#CDE7B0",
  "#A5D8FF",
  "#D3B8FF",
  "#F7A6FF",
];

// BAR chart with spring & bounce on X and Y
new Chart(document.getElementById("barChart").getContext("2d"), {
  type: "bar",
  data: {
    labels,
    datasets: [
      {
        label: "Annotations",
        data: scores,
        backgroundColor: labels.map(
          (_, i) => pastelPalette[i % pastelPalette.length],
        ),
        borderRadius: 12,
        barPercentage: 0.6,
      },
    ],
  },
  options: {
    responsive: true,
    animations: {
      y: {
        type: "number",
        easing: "easeOutQuart",
        duration: 2000,
        from: 1000,
      },
    },
    transitions: {
      active: {
        animation: {
          duration: 400,
          easing: "easeOutQuart",
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        bodyFont: { size: 15 },
        titleFont: { size: 16, weight: "600" },
        padding: 12,
        cornerRadius: 8,
      },
    },
    scales: {
      x: { ticks: { font: { size: 15 } }, grid: { display: false } },
      y: { ticks: { font: { size: 15 } }, grid: { color: "rgba(0,0,0,0.05)" } },
    },
  },
});

// DONUT chart with rotate+scale+bounce
new Chart(document.getElementById("pieChart").getContext("2d"), {
  type: "doughnut",
  data: {
    labels,
    datasets: [
      {
        data: scores,
        backgroundColor: labels.map(
          (_, i) => pastelPalette[i % pastelPalette.length],
        ),
      },
    ],
  },
  options: {
    cutout: "50%",
    responsive: true,
    animations: {
      animateRotate: {
        duration: 2000,
        easing: "easeOutQuart",
      },
    },
    plugins: {
      legend: {
        position: "right",
        labels: { font: { size: 15 }, boxWidth: 14, padding: 20 },
      },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
            const pct = ((ctx.raw / total) * 100).toFixed(1);
            return `${ctx.label}: ${ctx.raw} (${pct}%)`;
          },
        },
        bodyFont: { size: 15 },
        titleFont: { size: 16, weight: "600" },
        padding: 12,
        cornerRadius: 8,
      },
    },
    transitions: {
      active: { animation: { duration: 400, easing: "easeOutQuart" } },
    },
  },
});
