var start = new Date().getTime();
var elapsed = function() {
    return new Date().getTime() - start;
};
function logStatus(line) {
    // used by gfilter.js too
    document.getElementById("statusBox").innerHTML = line;
    console.log(elapsed() + ' ' + line);
}
function mainDone() {
    var spinner = document.getElementById('spinner');
    if(spinner)
        spinner.parentNode.removeChild(spinner);
}

(function () {
    var dropperFileInput = d3.select("#gfilterDropFiles");
    var dropperSelect = d3.select(window);
    var dropperViz = d3.select("#dropzone");
    var uiUpdateTitle = function(title) {
        d3.select("#fileName").text(" - " + title);
    };
    var uiFileHover = function() {
        dropperViz.classed("active", true);
    };
    var uiEndFileHover = function() {
        dropperViz.classed("active", false);
    };
    var readFiles = function (files) {
        var dataArray = files[0].data;
        uiUpdateTitle(files[0].name);
        handleData(dataArray);
    };
    var dropper = dropperSelect
        .call(dnd.dropper()
            .on("dragover", uiFileHover)
            .on("drop", uiEndFileHover)
            .on("read", readFiles)
            );
    
    dropperFileInput.on("change", function() {
        var file = this.files[0];
        dnd.read(file, function(error, data) {
            file.data = data;
            readFiles([file]);
        });
    });
    
    window.addEventListener("dragleave", function (e) {
        dropperViz.classed("active", false);
    });


    function getParameters() {
        var prmstr = window.location.search.substr(1);
        if (prmstr !== null && prmstr !== "") {
            if (prmstr.indexOf(':::') === -1)
                return transformToAssocArray(prmstr, "&", "=");
            else

                return transformToAssocArray(prmstr, "...", ":::");
        } else {
            return {};
        }
    }

    function transformToAssocArray(prmstr, enderStr, equalStr) {
        var params = {};
        var prmarr = prmstr.split(enderStr);
        for (var i = 0; i < prmarr.length; i++) {
            var pairs = prmarr[i].split(equalStr);
            params[pairs[0]] = decodeURIComponent(pairs[1]);
        }
        return params;
    }
    
    function error(line) {
        humane.error(line);
        console.error(line);
    }

    function handleData(data, params) {
        params = params || {};
        var preProcessCode = params['pre'];
        
        var vizType = params['viz'];
        var xprop = params['xprop'];
        var lineTypeProp = params['linetypeprop'];
        var multiPlot = params['multiplot']
        
        logStatus("Got all data, now analyzing");
        if(preProcessCode) {

            eval(preProcessCode);
        }
        if(data === null) {
            error("Failed to fetch CSV url");
            return;
        }
        if(!data.length) {
            error("Empty or invalid CSV from url");
            return;
        }
        var rootElement = document.getElementById('vizContainer');
        rootElement.innerHTML = '';
        switch(vizType) {
            case "plot":
                logStatus("Plotting");
                setTimeout(function() {
                    plotter.show(rootElement, data, xprop, lineTypeProp, multiPlot);
                });
                break;
            case "gfilter":
            /* falls through */
            default:
                gfilter(data, rootElement);
                break;
        }
    }
        
    
    function main() {
        var params = getParameters();
        var downloadUrl = params['dl'];
        var type = params['type'];
        
        function showProgress(dgetter) {
            logStatus("Downloading data");

            dgetter
                .on("load", function(data) {
                    handleData(data, params);
                })
                .on("progress", function() {
                    logStatus(d3.event.loaded + ' / ' + d3.event.total);
                })
                .get();
        }
        

        if (downloadUrl) {
            var dotLoc = downloadUrl.lastIndexOf('.');
            var extension = null;
            if(dotLoc != -1)
                extension = downloadUrl.substring(dotLoc);
            if(extension === '.json' || type === 'json') {
                showProgress(d3.json(downloadUrl));
            } else {
                showProgress(d3.csv(downloadUrl));
            }
        } else {
            mainDone();
        }
    }
    
    main();
})();