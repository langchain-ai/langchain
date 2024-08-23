// Sorry py folks, gotta be js for this one
const { checkBrokenLinks } = require("@langchain/scripts/check_broken_links");

checkBrokenLinks("docs", {
  timeout: 10000,
  retryFailed: true,
});
