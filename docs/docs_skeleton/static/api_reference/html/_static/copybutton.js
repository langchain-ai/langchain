// Localization support
const messages = {
  'en': {
    'copy': 'Copy',
    'copy_to_clipboard': 'Copy to clipboard',
    'copy_success': 'Copied!',
    'copy_failure': 'Failed to copy',
  },
  'es' : {
    'copy': 'Copiar',
    'copy_to_clipboard': 'Copiar al portapapeles',
    'copy_success': '¡Copiado!',
    'copy_failure': 'Error al copiar',
  },
  'de' : {
    'copy': 'Kopieren',
    'copy_to_clipboard': 'In die Zwischenablage kopieren',
    'copy_success': 'Kopiert!',
    'copy_failure': 'Fehler beim Kopieren',
  },
  'fr' : {
    'copy': 'Copier',
    'copy_to_clipboard': 'Copier dans le presse-papier',
    'copy_success': 'Copié !',
    'copy_failure': 'Échec de la copie',
  },
  'ru': {
    'copy': 'Скопировать',
    'copy_to_clipboard': 'Скопировать в буфер',
    'copy_success': 'Скопировано!',
    'copy_failure': 'Не удалось скопировать',
  },
  'zh-CN': {
    'copy': '复制',
    'copy_to_clipboard': '复制到剪贴板',
    'copy_success': '复制成功!',
    'copy_failure': '复制失败',
  },
  'it' : {
    'copy': 'Copiare',
    'copy_to_clipboard': 'Copiato negli appunti',
    'copy_success': 'Copiato!',
    'copy_failure': 'Errore durante la copia',
  }
}

let locale = 'en'
if( document.documentElement.lang !== undefined
    && messages[document.documentElement.lang] !== undefined ) {
  locale = document.documentElement.lang
}

let doc_url_root = DOCUMENTATION_OPTIONS.URL_ROOT;
if (doc_url_root == '#') {
    doc_url_root = '';
}

/**
 * SVG files for our copy buttons
 */
let iconCheck = `<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-check" width="44" height="44" viewBox="0 0 24 24" stroke-width="2" stroke="#22863a" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <title>${messages[locale]['copy_success']}</title>
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <path d="M5 12l5 5l10 -10" />
</svg>`

// If the user specified their own SVG use that, otherwise use the default
let iconCopy = ``;
if (!iconCopy) {
  iconCopy = `<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <title>${messages[locale]['copy_to_clipboard']}</title>
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <rect x="8" y="8" width="12" height="12" rx="2" />
  <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2" />
</svg>`
}

/**
 * Set up copy/paste for code blocks
 */

const runWhenDOMLoaded = cb => {
  if (document.readyState != 'loading') {
    cb()
  } else if (document.addEventListener) {
    document.addEventListener('DOMContentLoaded', cb)
  } else {
    document.attachEvent('onreadystatechange', function() {
      if (document.readyState == 'complete') cb()
    })
  }
}

const codeCellId = index => `codecell${index}`

// Clears selected text since ClipboardJS will select the text when copying
const clearSelection = () => {
  if (window.getSelection) {
    window.getSelection().removeAllRanges()
  } else if (document.selection) {
    document.selection.empty()
  }
}

// Changes tooltip text for a moment, then changes it back
// We want the timeout of our `success` class to be a bit shorter than the
// tooltip and icon change, so that we can hide the icon before changing back.
var timeoutIcon = 2000;
var timeoutSuccessClass = 1500;

const temporarilyChangeTooltip = (el, oldText, newText) => {
  el.setAttribute('data-tooltip', newText)
  el.classList.add('success')
  // Remove success a little bit sooner than we change the tooltip
  // So that we can use CSS to hide the copybutton first
  setTimeout(() => el.classList.remove('success'), timeoutSuccessClass)
  setTimeout(() => el.setAttribute('data-tooltip', oldText), timeoutIcon)
}

// Changes the copy button icon for two seconds, then changes it back
const temporarilyChangeIcon = (el) => {
  el.innerHTML = iconCheck;
  setTimeout(() => {el.innerHTML = iconCopy}, timeoutIcon)
}

const addCopyButtonToCodeCells = () => {
  // If ClipboardJS hasn't loaded, wait a bit and try again. This
  // happens because we load ClipboardJS asynchronously.
  if (window.ClipboardJS === undefined) {
    setTimeout(addCopyButtonToCodeCells, 250)
    return
  }

  // Add copybuttons to all of our code cells
  const COPYBUTTON_SELECTOR = 'div.highlight pre';
  const codeCells = document.querySelectorAll(COPYBUTTON_SELECTOR)
  codeCells.forEach((codeCell, index) => {
    const id = codeCellId(index)
    codeCell.setAttribute('id', id)

    const clipboardButton = id =>
    `<button class="copybtn o-tooltip--left" data-tooltip="${messages[locale]['copy']}" data-clipboard-target="#${id}">
      ${iconCopy}
    </button>`
    codeCell.insertAdjacentHTML('afterend', clipboardButton(id))
  })

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
 * Removes excluded text from a Node.
 *
 * @param {Node} target Node to filter.
 * @param {string} exclude CSS selector of nodes to exclude.
 * @returns {DOMString} Text from `target` with text removed.
 */
function filterText(target, exclude) {
    const clone = target.cloneNode(true);  // clone as to not modify the live DOM
    if (exclude) {
        // remove excluded nodes
        clone.querySelectorAll(exclude).forEach(node => node.remove());
    }
    return clone.innerText;
}

// Callback when a copy button is clicked. Will be passed the node that was clicked
// should then grab the text and replace pieces of text that shouldn't be used in output
function formatCopyText(textContent, copybuttonPromptText, isRegexp = false, onlyCopyPromptLines = true, removePrompts = true, copyEmptyLines = true, lineContinuationChar = "", hereDocDelim = "") {
    var regexp;
    var match;

    // Do we check for line continuation characters and "HERE-documents"?
    var useLineCont = !!lineContinuationChar
    var useHereDoc = !!hereDocDelim

    // create regexp to capture prompt and remaining line
    if (isRegexp) {
        regexp = new RegExp('^(' + copybuttonPromptText + ')(.*)')
    } else {
        regexp = new RegExp('^(' + escapeRegExp(copybuttonPromptText) + ')(.*)')
    }

    const outputLines = [];
    var promptFound = false;
    var gotLineCont = false;
    var gotHereDoc = false;
    const lineGotPrompt = [];
    for (const line of textContent.split('\n')) {
        match = line.match(regexp)
        if (match || gotLineCont || gotHereDoc) {
            promptFound = regexp.test(line)
            lineGotPrompt.push(promptFound)
            if (removePrompts && promptFound) {
                outputLines.push(match[2])
            } else {
                outputLines.push(line)
            }
            gotLineCont = line.endsWith(lineContinuationChar) & useLineCont
            if (line.includes(hereDocDelim) & useHereDoc)
                gotHereDoc = !gotHereDoc
        } else if (!onlyCopyPromptLines) {
            outputLines.push(line)
        } else if (copyEmptyLines && line.trim() === '') {
            outputLines.push(line)
        }
    }

    // If no lines with the prompt were found then just use original lines
    if (lineGotPrompt.some(v => v === true)) {
        textContent = outputLines.join('\n');
    }

    // Remove a trailing newline to avoid auto-running when pasting
    if (textContent.endsWith("\n")) {
        textContent = textContent.slice(0, -1)
    }
    return textContent
}


var copyTargetText = (trigger) => {
  var target = document.querySelector(trigger.attributes['data-clipboard-target'].value);

  // get filtered text
  let exclude = '.linenos';

  let text = filterText(target, exclude);
  return formatCopyText(text, '', false, true, true, true, '', '')
}

  // Initialize with a callback so we can modify the text before copy
  const clipboard = new ClipboardJS('.copybtn', {text: copyTargetText})

  // Update UI with error/success messages
  clipboard.on('success', event => {
    clearSelection()
    temporarilyChangeTooltip(event.trigger, messages[locale]['copy'], messages[locale]['copy_success'])
    temporarilyChangeIcon(event.trigger)
  })

  clipboard.on('error', event => {
    temporarilyChangeTooltip(event.trigger, messages[locale]['copy'], messages[locale]['copy_failure'])
  })
}

runWhenDOMLoaded(addCopyButtonToCodeCells)