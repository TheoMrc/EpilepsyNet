document.addEventListener("DOMContentLoaded", () => {
  const pageBlur = document.getElementById("page-blur");
  if (pageBlur) {
    if (pageBlur.style.display !== "none") {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
  }
});
