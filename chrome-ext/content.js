// This part reacts to clicking the icon & open last visited tab
// chrome.runtime.onMessage.addListener(
//     function(request, sender, sendResponse) {
//         if( request.message === "clicked_browser_action" ) {
//             var firstHref = $("a[href^='http']").eq(0).attr("href");
//             console.log(firstHref);
//             chrome.runtime.sendMessage({"message": "open_new_tab", "url": firstHref});
//         }
//     }
// );

// Previous dom, that we want to track.
var prevDOM = null;
var buttonElement = document.createElement("BUTTON");
var buttonContent = document.createTextNode("Buy");
buttonElement.setAttribute("id", "app_buy_button");
buttonElement.setAttribute("class", "app_button")
buttonElement.appendChild(buttonContent);

// Mouse listener for any move event on the current document.
document.addEventListener('mousemove', function (e) {
    var srcElement = e.srcElement;

    // Lets check if our underlying element is a DIV.
    if (srcElement.nodeName == 'IMG') {
        // For NPE checking, we check safely. We need to remove the class name
        // Since we will be styling the new one after.
        if (prevDOM != null) {
            // Remove buttonElement as child of previous element
            // the parentElement will be the one we added
            prevDOM.parentElement.removeChild(buttonElement);
        }
        
        if (srcElement.parentElement.className != "app_wrap_div") {
            console.log('HERE');
            // Need to add an extra DIV
            var oldParentDiv = srcElement.parentElement;
            var newParentDiv = document.createElement("DIV");
            newParentDiv.setAttribute("class", "app_wrap_div");
            newParentDiv.appendChild(srcElement);
            newParentDiv.appendChild(buttonElement);
            oldParentDiv.appendChild(newParentDiv);
        } else {
            // We have already added a DIV... directly append child
            srcElement.parentElement.appendChild(buttonElement)
        }
 
        // The current element is now the previous. So we can remove the class
        // during the next iteration.
        prevDOM = srcElement;
        console.info(srcElement.currentSrc);
        console.dir(srcElement);
    }
}, false);
