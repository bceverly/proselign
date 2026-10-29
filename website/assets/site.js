// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryan C. Everly
// Adds a "Copy" button to every code block.  DOM is built with
// createElement/textContent only -- no HTML strings (see tests/lucky13).
(function () {
  "use strict";
  document.querySelectorAll("pre.code").forEach(function (pre) {
    if (!navigator.clipboard) {
      return;
    }
    var wrapper = document.createElement("div");
    wrapper.className = "code-block";
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.textContent.replace(/\n$/, "")).then(function () {
        button.textContent = "Copied";
        setTimeout(function () { button.textContent = "Copy"; }, 1500);
      });
    });
    wrapper.appendChild(button);
  });
})();
