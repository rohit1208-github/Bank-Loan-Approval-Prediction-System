/* global dc */
/* global dnd */
/* global crossfilter */
var gfilter = function (data, rootElement) {
    gfilter.init(data, rootElement);
};

gfilter.className = "gfilter";
gfilter.width = 470;
gfilter.height = 300;


gfilter.addData = function (data) {
    gfilter.crossfilter.add(data);
    dc.redrawAll();
};

gfilter.humanizePercent = function(x) {
    return x.toFixed(2).replace(/\.?0*$/,'');
}

gfilter.isNumeric = function (n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}
    
gfilter.isNumericArray = function(seq) {
    for(var i = 0; i < seq.length; i++) {
        if(gfilter.isNumeric(seq[i]))
            return true;
    }
};

gfilter.init = function (data, rootElement) {
    var parseDate = function(obj) {
        var iso = d3.time.format.utc("%Y-%m-%dT%H:%M:%S");
        return iso.parse(obj);
    };
    
    var isDate = function(obj) {
        return parseDate(obj) != null;
    };

    var addDiv = function (id) {
        var div = document.createElement("div");
        div.id = id;
        div.className = gfilter.className;
        rootElement.appendChild(div);
        
        return div;
    }

    function createFilterDiv(div) {
        var controls = document.createElement("div");
        controls.className = 'filter';
        div.appendChild(controls);
    }
    
    function createChartDiv(propName) {
        var escapedName = propName.replace(' ', '').replace('.', '');
        var chartId = "chart-hist-" + escapedName;
        var chartDiv = addDiv(chartId);
        createFilterDiv(chartDiv);
        return chartDiv;
    }

    var addText = function (text, parentDiv, cls) {
        var textNode = document.createTextNode(text);
        var line = document.createElement("div");
        line.className = cls;
        line.appendChild(textNode);
        parentDiv.appendChild(line);
    }

    var createDataWidget = function () {
        var dataTableId = "dataTable";
        var tableDiv = addDiv(dataTableId);
        d3.select(tableDiv).classed("table", true);
        var table = dc.dataTable("#" + dataTableId);
        var getFirstParam = function (d) {
            return d[params[0]];
        };
        var tableDim = ndx.dimension(getFirstParam);

        table
            .width(800)
            .height(600)
            .dimension(tableDim)
            .group(getFirstParam)
            .showGroups(false)
            .size(10)
            .columns(params)
    };
    
    var complaintsDiv = addDiv("complaints");
    var params = Object.keys(data[0]);
    var ndx = crossfilter(data);
    gfilter.crossfilter = ndx;
    gfilter.dimensions = {};

    var failedColumns = [];
    
    var createRowCounter = function() {
        var rowCounterId = 'rowCounter'
        addDiv(rowCounterId);
        var rowCounter = dc.numberDisplay ('#' + rowCounterId);
        rowCounter
            .dimension(ndx)
            .group(ndx.groupAll())
            .formatNumber(function() {
                var selectedCount = rowCounter.group().value();
                var total = rowCounter.dimension().size();
                if(selectedCount == total) {
                    return 'All <strong>' + total + '</strong> records shown. Click on the graphs to apply filters.'
                }
                var resetButton = '<a href="javascript:dc.filterAll(); dc.renderAll();">Reset All</a>';
                var percent = gfilter.humanizePercent(100 * selectedCount / total);
                return 'Selected ' + percent + '% (' + selectedCount + ' out of ' + total + ' records) | ' + resetButton;
        });
    }

    var createDateHistogram = function (propName) {
        var chartDiv = createChartDiv(propName);
        addText(propName, chartDiv, "chartTitle");
        data.forEach(function (d) {
            d[propName] = parseDate(d[propName]);
        });
        var valueFunc = function(d) {
            return d[propName];
        };
        var minMax = d3.extent(data, valueFunc);
        var min = minMax[0];
        var max = minMax[1];
        var span = max - min;

        var lastBarSize = 0;
        var barCount = 30;

        // avoid very thin lines and a barcode-like histogram
        lastBarSize = span / barCount;
        var roundToHistogramBar = function (d) {
            if (isNaN(d) || d === "")
                d = NaN;
            if (d == max)

                d = max - lastBarSize;
            var res = new Date(min.getTime() + span * Math.floor(barCount * (d - min) / span) / barCount);
            return res;
        };

        var dimDate = ndx.dimension(valueFunc);
        var barChart = dc.barChart(chartDiv);
        barChart
            .width(gfilter.width).height(gfilter.height)
            .controlsUseVisibility(true)
            .dimension(dimDate)
            .group(dimDate.group(roundToHistogramBar))
            .x(d3.time.scale.utc().domain([min, max]))
            .elasticY(true)
            .yAxis().ticks(2);
        barChart.xUnits(function () { return barCount; })
    };
        
    var createHistogram = function (propName) {
        var chartDiv = createChartDiv(propName);
        addText(propName, chartDiv, "chartTitle");
        var numericValue = function (d) {
            if (d[propName] === "")
                return NaN;
            else
                return +d[propName];
        };
        var minMax = d3.extent(data, numericValue);
        var min = minMax[0];
        var max = minMax[1];
        var span = max - min;
        numericValue = function (d) {
            if (d[propName] === "")
                return min - max;
            else
                return +d[propName];
        };
        var dimNumeric = ndx.dimension(numericValue);
        gfilter.dimensions[propName] = dimNumeric;
        var countGroup;
        var lastBarSize = 0;
        var barCount = 30;
        if (5 < span && span < 60) {
        }

        lastBarSize = span / barCount;
        var roundToHistogramBar = function (d) {
            if (isNaN(d) || d === "")
                d = NaN;
            if (d == max)
                d = max - lastBarSize;
            var res = min + span * Math.floor(barCount * (d - min) / span) / barCount;
            return res;
        };
        countGroup = dimNumeric.group(roundToHistogramBar);
        gfilter.group = countGroup;
        var barChart = dc.barChart(chartDiv);
        barChart.xUnits(function () { return barCount; });

        barChart
            .width(gfilter.width).height(gfilter.height)
            .dimension(dimNumeric)
            .group(countGroup)
            .x(d3.scale.linear().domain([min - lastBarSize, max + lastBarSize]).rangeRound([0, 500]))
            .elasticY(true)
            .controlsUseVisibility(true);
        barChart.yAxis().ticks(2);
    }

    var createRowChart = function (propName) {
        var chartDiv = createChartDiv(propName);
        addText(propName, chartDiv, "chartTitle");
        var dim = ndx.dimension(function (d) {

            return "" + d[propName];
        });
        var group = dim.group().reduceCount();
        var rowChart = dc.rowChart(chartDiv);
        rowChart
            .width(gfilter.width)
            .height(gfilter.height)
            .controlsUseVisibility(true)
            .dimension(dim)
            .group(group)
            .elasticX(true);
    }

    function getTopN(dataArr, n) {
        var counts = {};

        for(var i = 0; i< dataArr.length; i++) {
            var val = dataArr[i];
            counts[val] = counts[val] ? counts[val] + 1 : 1;
        };

        var keysSorted = Object.keys(counts).sort(function(a, b) {
            return counts[a] - counts[b];
        });

        var topKeysArray = keysSorted.slice(-n);
        var othersCount = dataArr.length;
        var topSingleCount = 0;
        topKeysArray.map(function(key) {
            var keyCount = counts[key];
            othersCount = othersCount - keyCount;
            if(keyCount > topSingleCount)
                topSingleCount = keyCount;
        });
        return {
            "topKeysArray": topKeysArray,
            "othersCount": othersCount,
            "topSingleCount": topSingleCount
        }
    }

    function createRowChartWithOthers(propName, data) {
        var values = data.map(function(d) {
            return d[propName];
        });
        var topNInfo = getTopN(values, 10);
        if (topNInfo.topSingleCount * 5 < topNInfo.othersCount) {

            return false;
        }
        
        var topItemsList = topNInfo.topKeysArray;
        var topItemsObj = {};
        topItemsList.map(function(val, index) {
            topItemsObj[val] = true;
        });

        var newDim = ndx.dimension(function (d) {

            var key = d[propName];
            if (topItemsObj[key])
                return "" + key;
            else
                return "[...other...]";
        });

        var chartDiv = createChartDiv(propName);
        addText(propName, chartDiv, "chartTitle");
        var group = newDim.group().reduceCount();
        var rowChart = dc.rowChart(chartDiv);
        rowChart
            .width(gfilter.width)
            .height(gfilter.height)
            .controlsUseVisibility(true)
            .dimension(newDim)
            .group(group)
            .elasticX(true);

        return true;
    }

    function showColumn() {
        var allowedTypes = ["boolean", "number", "string"];
        var propFirstType = typeof data[0][propName];
        if (allowedTypes.indexOf(propFirstType) === -1) {
            // type not supported
            failedColumns.push(propName);
            return;
        }

        var uniques = d3.map(data, function (d) { return d[propName] });
        var uniqueCount = uniques.size();
        if (uniqueCount < 2) {
            failedColumns.push(propName);
            return;
        } else if (uniqueCount < 6) {
            createRowChart(propName);
        } else if (gfilter.isNumericArray(uniques.keys())) {
            createHistogram(propName);
        } else if (uniqueCount < 21) {
            createRowChart(propName);
        } else if (isDate(data[0][propName])) {
            createDateHistogram(propName);
        } else if (createRowChartWithOthers(propName, data)) {
        } else {
            failedColumns.push(propName);
        }
    }
    
    var paramIndex = 0;
    var propName;
    function analyzeAndShowColumns() {
        if (paramIndex < params.length) {
            propName = params[paramIndex];
            logStatus("Analyzing " + propName);
            paramIndex++;

            funcArray.unshift(showColumn, analyzeAndShowColumns);
        }
    }
    
    function finish() {
        if (failedColumns.length > 0) {
            var complaintText = "Not creating chart for the column(s): " + failedColumns.join(", ");
            if(typeof logStatus === "undefined")
                addText(complaintText, complaintsDiv, "complaint");
            else
                logStatus(complaintText);
        }

        dc.renderAll();
        
        if(typeof mainDone !== "undefined") {
            mainDone();
        }
    }
    
    var funcArray = [];
    function setTimeoutCallForEach() {
        if(funcArray.length) {
            var nextFunc = funcArray.shift();
            nextFunc();
            setTimeout(setTimeoutCallForEach);
        }
    }

    function main() {
        funcArray = [
            analyzeAndShowColumns,
            createRowCounter,
            createDataWidget,
            finish
        ];
        setTimeout(setTimeoutCallForEach);
    }
    
    main();
};


