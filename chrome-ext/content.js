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

var oldParentDiv, newParentDiv, oldParentChildren, imageIndex, nextChild;
var prevDOM = null;  // Previous dom, that we want to track.
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

        oldParentDiv = srcElement.parentElement;
        oldParentChildren = [...oldParentDiv.children];
        imageIndex = oldParentChildren.indexOf(srcElement);
        
        if (srcElement.parentElement.className != "app_wrap_div") {
            if (oldParentChildren.length > imageIndex + 1) {
                // If more children exist after element...
                nextChild = oldParentChildren[imageIndex + 1];  // get next child
            }
            // Need to add an extra DIV
            newParentDiv = document.createElement("DIV");
            newParentDiv.setAttribute("class", "app_wrap_div");
            newParentDiv.appendChild(srcElement);
            newParentDiv.appendChild(buttonElement);
            
            if (oldParentChildren.length > imageIndex + 1) {
                oldParentDiv.insertBefore(newParentDiv, nextChild);
            } else {  // Either the last or only child
                oldParentDiv.appendChild(newParentDiv);
            }
        } else {
            // We have already added a DIV... directly add child to DIV
            srcElement.parentElement.appendChild(buttonElement)
        }
 
        // The current element is now the previous. So we can remove the class
        // during the next iteration.
        prevDOM = srcElement;
    }
}, false);
