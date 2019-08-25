// Send the active tab to the listener
// chrome.browserAction.onClicked.addListener(function(tab) {
//     chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
//         var activeTab = tabs[0];
//         chrome.tabs.sendMessage(activeTab.id, 
//                                 {"message": "clicked_browser_action"});
//     });
// });

// Actually open the tab
// chrome.runtime.onMessage.addListener(
//     function(request, sender, sendResponse) {
//         if( request.message === "open_new_tab" ) {
//             chrome.tabs.create({"url": request.url});
//         }
//     }
// );