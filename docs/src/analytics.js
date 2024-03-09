export default function gtag(...args) {
  if (window.gtagFeedback) {
    window.gtagFeedback(...args);
  }
}
