// Download image
chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if ( request.message === "download_image" ) {
            var url = request.url;
            var code = request.code;
            // TODO: do something with code (sent to server)
        }
    }
);