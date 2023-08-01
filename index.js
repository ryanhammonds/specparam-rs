import {get_array, get_vec_pointer, simulate, fit} from "specparam-rs";

function fill_pointer(arr, values) {
    for (let i = 0; i < values.length; i++){
        arr[i] = values[i];
    }
}

function get_params(){
    console.log("get_params");

    var ap_params = [];

    if (typeof sliderInputExp0 !== 'undefined'){
        ap_params.push(parseFloat(sliderInputKnee0.value));
        ap_params.push(parseFloat(sliderInputExp0.value));
        ap_params.push(parseFloat(Math.pow(10, sliderInputOffset0.value)));
        ap_params.push(parseFloat(sliderInputKnee1.value));
        ap_params.push(parseFloat(sliderInputExp1.value));
        ap_params.push(parseFloat(Math.pow(10, sliderInputOffset1.value)));
    } else if (typeof sliderInputExp !== 'undefined'){

        if (typeof sliderInputKnee !== 'undefined'){
            ap_params.push(parseFloat(sliderInputKnee.value));
            ap_params.push(parseFloat(sliderInputExp.value));
            ap_params.push(parseFloat(Math.pow(10, sliderInputOffset.value)));
        } else{
            ap_params.push(parseFloat(sliderInputExp.value));
            ap_params.push(parseFloat(sliderInputOffset.value));
        };
    };

    var pe_params = [];

    if (typeof sliderOsc0CF !== 'undefined'){

        if (noscin.value > 0){
            let oscdivs = document.getElementById('pediv').children[2].children[0].children;
            for (let i=0; i < oscdivs.length; i++){
                if (oscdivs[i].localName == 'input'){
                    pe_params.push(parseFloat(oscdivs[i].value));
                }
            };
        }
    };

    let ap_params_ptr = get_vec_pointer(ap_params.length);
    let ap_params_arr = get_array(ap_params_ptr, ap_params.length);
    fill_pointer(ap_params_arr, ap_params);

    let pe_params_ptr = get_vec_pointer(pe_params.length);
    let pe_params_arr = get_array(pe_params_ptr, pe_params.length);
    fill_pointer(pe_params_arr, pe_params);

    let f_res = document.getElementById("fresin").value;

    let _ap_mode = 0.0;
    if (selectModel.value == 'Linear') {
        _ap_mode = 0.0;
    } else if (selectModel.value == 'Lorentzian') {
        _ap_mode = 1.0;
    } else {
        _ap_mode = 2.0;
    };

    let _powers = simulate(
        ap_params_ptr,
        ap_params.length,
        pe_params_ptr,
        pe_params.length,
        0.01,
        sliderPeakWidthLower.value,
        sliderPeakWidthUpper.value,
        sliderMaxPeaks.value,
        sliderMinHeight.value,
        sliderPeakThresh.value,
        _ap_mode,
        f_res,
    );

    let third = _powers.length / 3;
    let freqs = _powers.slice(0, third);
    let powers = _powers.slice(third, 2*third);
    let powers_fit = _powers.slice(2*third, 3*third);

    // Plot
    var graphDiv = document.getElementById("graph");

    var trace1 = {
        x: freqs,
        y: powers,
        mode: 'lines',
        name: 'original'
    };

    var trace2 = {
        x: freqs,
        y: powers_fit,
        mode: 'lines',
        line: {
            dash: 'dash'
        },
        name: 'fit',
    };

    var layout = {
        xaxis: {
            type: 'log',
            //yaxis: {range: [.1, 10]}
        },
        yaxis: {
            type: 'log',
            //range: [-5, 4]
        }
    };
    var data = [trace1, trace2];
    Plotly.newPlot(graphDiv, data, layout);
};

function peOptions(val) {
    let osc_div = document.createElement('div');

    const pediv = document.getElementById('pediv');
    while (pediv.childNodes.length > 4) {
        pediv.removeChild(pediv.lastChild);
    }
    console.log(pediv);
    if (val > 0){
        let osc_str = "<a>";
        for (let i=0; i < val; i++){
            osc_str += `
              Oscillation ${i}
                <br>
                Center Frequency <span id="osc${i}cf"></span>
                <input type="range" min="1" max="100.0" value="${(i+1)*10}" step="0.1" class="slider"
                id="sliderOsc${i}CF" autocomplete="off">

                Height <span id="osc${i}PW"></span>
                <input type="range" min="0" max="10.0" value="${3.0 / (i+1)}" step="0.1" class="slider"
                id="sliderOsc${i}PW"  autocomplete="off">

                Bandwith <span id="osc${i}bw"></span>
                <input type="range" min=".1" max="10.0" value="1" step="0.1" class="slider"
                id="sliderOsc${i}BW"  autocomplete="off">
            `;
        }
        osc_str += "</a>";
        osc_div.innerHTML = osc_str;

        for (let j=0; j < osc_div.children[0].children.length; j++){
            if (osc_div.children[0].children[j].localName == "input"){
                osc_div.children[0].children[j].onmouseup = function() {
                    osc_div.children[0].children[j-1].innerHTML = this.value;
                    get_params();
                };
            };
        }

        document.getElementById('pediv').appendChild(osc_div);
    };
    get_params();
};

function apOptions(){
    var ap_mode = document.getElementById("selectboxap");
    if (ap_mode.value == 'Linear') {
        const linear_div = document.createElement('div');
        linear_div.innerHTML = `
          <a>
            Exponent
            <span id="expOutput">2.0</span>
            <input type="range" min="0.0" max="4.0" value="2.0" step="0.1" class="slider"
              id="sliderInputExp" autocomplete="off">

            Offset
            <span id="offOutput">0.0</span>
            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"
              id="sliderInputOffset" autocomplete="off">
          </a>
        `;

        linear_div.children[0].children.sliderInputExp.onmouseup = function() {
            linear_div.children[0].children.expOutput.innerHTML = this.value;
            get_params();
        }

        linear_div.children[0].children.sliderInputOffset.onmouseup = function() {
            linear_div.children[0].children.offOutput.innerHTML = this.value;
            get_params();
        }

        const ap = document.getElementById('apdiv')
        while (ap.childNodes.length > 2) {
            ap.removeChild(ap.lastChild);
        }
        ap.appendChild(linear_div);

    } else if (ap_mode.value == 'Lorentzian') {
        const lorentzian_div = document.createElement('div');
        lorentzian_div.innerHTML = `
        <a>
            Knee Frequency
            <span id="kneeOutput">10.0</span>
            <input type="range" min="1.0" max="100.0" value="10.0" step="0.1" class="slider"
              id="sliderInputKnee" autocomplete="off"">

            Exponent
            <span id="expOutput">2.0</span>
            <input type="range" min="0.0" max="4.0" value="2.0" step="0.1" class="slider"
              id="sliderInputExp" autocomplete="off"">

            Offset
            <span id="offOutput">0.0</span>
            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"
              id="sliderInputOffset" autocomplete="off"">
        </a>
        `;

        lorentzian_div.children[0].children.sliderInputKnee.onmouseup = function() {
            lorentzian_div.children[0].children.kneeOutput.innerHTML = this.value;
            get_params();
        }

        lorentzian_div.children[0].children.sliderInputExp.onmouseup = function() {
            lorentzian_div.children[0].children.expOutput.innerHTML = this.value;
            get_params();
        }

        lorentzian_div.children[0].children.sliderInputOffset.onmouseup = function() {
            lorentzian_div.children[0].children.offOutput.innerHTML = this.value;
            get_params();
        }

        const ap = document.getElementById('apdiv')
        while (ap.childNodes.length > 2) {
            ap.removeChild(ap.lastChild);
        }
        ap.appendChild(lorentzian_div);
    } else {
        const double_lorentzian_div = document.createElement('div');
        double_lorentzian_div.innerHTML = `
        <a>
            Knee Frequency 0
            <span id="kneeOutput0">10.0</span>
            <input type="range" min="1.0" max="100.0" value="10.0" step="0.1" class="slider"
              id="sliderInputKnee0" autocomplete="off"">

            Exponent 0
            <span id="expOutput0">3.0</span>
            <input type="range" min="0.0" max="4.0" value="3.0" step="0.1" class="slider"
              id="sliderInputExp0" autocomplete="off"">

            Offset 0
            <span id="offOutput0">0.0</span>
            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"
              id="sliderInputOffset0" autocomplete="off"">

            Knee Frequency 1
            <span id="kneeOutput1">50.0</span>
            <input type="range" min="1.0" max="100.0" value="50.0" step="0.1" class="slider"
            id="sliderInputKnee1" autocomplete="off"">

            Exponent 1
            <span id="expOutput1">0.5</span>
            <input type="range" min="0.0" max="4.0" value="0.5" step="0.1" class="slider"
            id="sliderInputExp1" autocomplete="off"">

            Offset 1
            <span id="offOutput1">0.0</span>
            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"
            id="sliderInputOffset1" autocomplete="off"">
        </a>
        `;

        double_lorentzian_div.children[0].children.sliderInputKnee0.onmouseup = function() {
            double_lorentzian_div.children[0].children.kneeOutput0.innerHTML = this.value;
            get_params();
        }

        double_lorentzian_div.children[0].children.sliderInputExp0.onmouseup = function() {
            double_lorentzian_div.children[0].children.expOutput0.innerHTML = this.value;
            get_params();
        }

        double_lorentzian_div.children[0].children.sliderInputOffset0.onmouseup = function() {
            double_lorentzian_div.children[0].children.offOutput0.innerHTML = this.value;
            get_params();
        }

        double_lorentzian_div.children[0].children.sliderInputKnee1.onmouseup = function() {
            double_lorentzian_div.children[0].children.kneeOutput1.innerHTML = this.value;
            get_params();
        }

        double_lorentzian_div.children[0].children.sliderInputExp1.onmouseup = function() {
            double_lorentzian_div.children[0].children.expOutput1.innerHTML = this.value;
            get_params();
        }

        double_lorentzian_div.children[0].children.sliderInputOffset1.onmouseup = function() {
            double_lorentzian_div.children[0].children.offOutput1.innerHTML = this.value;
            get_params();
        }

        const ap = document.getElementById('apdiv')
        while (ap.childNodes.length > 2) {
            ap.removeChild(ap.lastChild);
        }
        ap.appendChild(double_lorentzian_div);
    }

    get_params();
};


function modelOptions(){
    let fitdiv = document.getElementById("fitdiv")

    fitdiv.innerHTML = `
        <a>
            Peak Width Lower Bound
            <span id="peakWidthLower">0.0</span>
            <input type="range" min="0.0" max="15.0" value="0.0" step="0.1" class="slider"
              id="sliderPeakWidthLower" autocomplete="off"">

            Peak Width Upper Bound
            <span id="peakWidthUpper">12.0</span>
            <input type="range" min="0.0" max="20.0" value="12.0" step="0.1" class="slider"
              id="sliderPeakWidthUpper" autocomplete="off"">

            Max Peaks
            <span id="maxPeaks">0.0</span>
            <input type="range" min="0.0" max="10.0" value="0.0" step="1.0" class="slider"
              id="sliderMaxPeaks" autocomplete="off"">

            Min Peak Height
            <span id="minHeight">0.0</span>
            <input type="range" min="0.0" max="10.0" value="0.0" step="0.1" class="slider"
              id="sliderMinHeight" autocomplete="off"">

            Peak Threshold
            <span id="peakThresh">2.0</span>
            <input type="range" min="0.0" max="6.0" value="2.0" step="0.1" class="slider"
              id="sliderPeakThresh" autocomplete="off"">

            Aperiodic Mode
            <select id="selectModel" autocomplete="off">
              <option value="Linear" selected="selected">Linear</option>
              <option value="Lorentzian">Lorentzian</option>
              <option value="Double Lorentzian">Double Lorentzian</option>
        </a>
        `;


    fitdiv.children[0].children.sliderPeakWidthLower.onmouseup = function() {
        fitdiv.children[0].children.peakWidthLower.innerHTML = this.value;
        get_params();
    }

    fitdiv.children[0].children.sliderPeakWidthUpper.onmouseup = function() {
        fitdiv.children[0].children.peakWidthUpper.innerHTML = this.value;
        get_params();
    }

    fitdiv.children[0].children.sliderMaxPeaks.onmouseup = function() {
        fitdiv.children[0].children.maxPeaks.innerHTML = this.value;
        get_params();
    }

    fitdiv.children[0].children.sliderMinHeight.onmouseup = function() {
        fitdiv.children[0].children.minHeight.innerHTML = this.value;
        get_params();
    }

    fitdiv.children[0].children.sliderPeakThresh.onmouseup = function() {
        fitdiv.children[0].children.peakThresh.innerHTML = this.value;
        get_params();
    }

    document.getElementById("selectModel").addEventListener("change", get_params);
};

modelOptions();

document.getElementById("selectboxap").addEventListener("change", apOptions);

document.getElementById("noscin").onmouseup = function() {
    peOptions(this.value);
};

document.getElementById("fresin").onmouseup = function() {
    document.getElementById("fres").innerHTML = this.value;
    get_params();
};
