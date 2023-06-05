try {
  var session = window.sessionStorage || {};
} catch (e) {
  var session = {};
}

window.addEventListener("DOMContentLoaded", () => {
  const allTabs = document.querySelectorAll('.sphinx-tabs-tab');
  const tabLists = document.querySelectorAll('[role="tablist"]');

  allTabs.forEach(tab => {
    tab.addEventListener("click", changeTabs);
  });

  tabLists.forEach(tabList => {
    tabList.addEventListener("keydown", keyTabs);
  });

  // Restore group tab selection from session
  const lastSelected = session.getItem('sphinx-tabs-last-selected');
  if (lastSelected != null) selectNamedTabs(lastSelected);
});

/**
 * Key focus left and right between sibling elements using arrows
 * @param  {Node} e the element in focus when key was pressed
 */
function keyTabs(e) {
    const tab = e.target;
    let nextTab = null;
    if (e.keyCode === 39 || e.keyCode === 37) {
      tab.setAttribute("tabindex", -1);
      // Move right
      if (e.keyCode === 39) {
        nextTab = tab.nextElementSibling;
        if (nextTab === null) {
          nextTab = tab.parentNode.firstElementChild;
        }
      // Move left
      } else if (e.keyCode === 37) {
        nextTab = tab.previousElementSibling;
        if (nextTab === null) {
          nextTab = tab.parentNode.lastElementChild;
        }
      }
    }

    if (nextTab !== null) {
      nextTab.setAttribute("tabindex", 0);
      nextTab.focus();
    }
}

/**
 * Select or deselect clicked tab. If a group tab
 * is selected, also select tab in other tabLists.
 * @param  {Node} e the element that was clicked
 */
function changeTabs(e) {
  // Use this instead of the element that was clicked, in case it's a child
  const notSelected = this.getAttribute("aria-selected") === "false";
  const positionBefore = this.parentNode.getBoundingClientRect().top;
  const notClosable = !this.parentNode.classList.contains("closeable");

  deselectTabList(this);

  if (notSelected || notClosable) {
    selectTab(this);
    const name = this.getAttribute("name");
    selectNamedTabs(name, this.id);

    if (this.classList.contains("group-tab")) {
      // Persist during session
      session.setItem('sphinx-tabs-last-selected', name);
    }
  }

  const positionAfter = this.parentNode.getBoundingClientRect().top;
  const positionDelta = positionAfter - positionBefore;
  // Scroll to offset content resizing
  window.scrollTo(0, window.scrollY + positionDelta);
}

/**
 * Select tab and show associated panel.
 * @param  {Node} tab tab to select
 */
function selectTab(tab) {
  tab.setAttribute("aria-selected", true);

  // Show the associated panel
  document
    .getElementById(tab.getAttribute("aria-controls"))
    .removeAttribute("hidden");
}

/**
 * Hide the panels associated with all tabs within the
 * tablist containing this tab.
 * @param  {Node} tab a tab within the tablist to deselect
 */
function deselectTabList(tab) {
  const parent = tab.parentNode;
  const grandparent = parent.parentNode;

  Array.from(parent.children)
  .forEach(t => t.setAttribute("aria-selected", false));

  Array.from(grandparent.children)
    .slice(1)  // Skip tablist
    .forEach(panel => panel.setAttribute("hidden", true));
}

/**
 * Select grouped tabs with the same name, but no the tab
 * with the given id.
 * @param  {Node} name name of grouped tab to be selected
 * @param  {Node} clickedId id of clicked tab
 */
function selectNamedTabs(name, clickedId=null) {
  const groupedTabs = document.querySelectorAll(`.sphinx-tabs-tab[name="${name}"]`);
  const tabLists = Array.from(groupedTabs).map(tab => tab.parentNode);

  tabLists
    .forEach(tabList => {
      // Don't want to change the tabList containing the clicked tab
      const clickedTab = tabList.querySelector(`[id="${clickedId}"]`);
      if (clickedTab === null ) {
        // Select first tab with matching name
        const tab = tabList.querySelector(`.sphinx-tabs-tab[name="${name}"]`);
        deselectTabList(tab);
        selectTab(tab);
      }
    })
}

if (typeof exports === 'undefined') {
  exports = {};
}

exports.keyTabs = keyTabs;
exports.changeTabs = changeTabs;
exports.selectTab = selectTab;
exports.deselectTabList = deselectTabList;
exports.selectNamedTabs = selectNamedTabs;
