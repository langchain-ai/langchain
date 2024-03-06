export default function gtag(...args) {
  if (window.gtag) {
    window.gtag(...args);
  }
}
