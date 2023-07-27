import {get_array, get_vec_pointer, simulate, fit} from "specparam-rs";

function fill_pointer(arr, values) {
    for (let i = 0; i < values.length; i++){
        arr[i] = values[i];
    }
}

function get_params(){
    console.log("get_params");

    var ap_params = [];

    if (typeof sliderInputExp !== 'undefined'){

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

    console.log(ap_params);
    console.log(pe_params);

    let ap_params_ptr = get_vec_pointer(ap_params.length);
    let ap_params_arr = get_array(ap_params_ptr, ap_params.length);
    fill_pointer(ap_params_arr, ap_params);

    let pe_params_ptr = get_vec_pointer(pe_params.length);
    let pe_params_arr = get_array(pe_params_ptr, pe_params.length);
    fill_pointer(pe_params_arr, pe_params);

    // console.log(selectModel.value);
    // console.log(sliderPeakWidthLower.value);
    // console.log(sliderPeakWidthUpper.value);
    // console.log(sliderMaxPeaks.value);
    // console.log(sliderPeakThresh.value);

    let _ap_mode = false;
    if (selectModel.value == 'Linear') {
        _ap_mode = true;
    };

    console.log(sliderMaxPeaks.value);
    let _powers = simulate(
        ap_params_ptr,
        ap_params.length,
        pe_params_ptr,
        pe_params.length,
        0.1,
        sliderPeakWidthLower.value,
        sliderPeakWidthUpper.value,
        sliderMaxPeaks.value,
        sliderMinHeight.value,
        sliderPeakThresh.value,
        _ap_mode
    );

    let powers = _powers.slice(0, 100);
    let freqs = Array.from({length: 100}, (_, i) => i + 1)
    let powers_fit = _powers.slice(100);

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
            range: [-5, 4]
        }
    };
    var data = [trace1, trace2];
    Plotly.newPlot(graphDiv, data, layout);
};

function peOptions(val) {
    let osc_div = document.createElement('div');

    pediv = document.getElementById('pediv');
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
                <input type="range" min="0" max="10.0" value="${4.0 / (i+1)}" step="0.1" class="slider"
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

        apdiv = document.getElementById('apdiv')
        while (apdiv.childNodes.length > 2) {
            apdiv.removeChild(apdiv.lastChild);
        }
        apdiv.appendChild(linear_div);

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

        apdiv = document.getElementById('apdiv')
        while (apdiv.childNodes.length > 2) {
            apdiv.removeChild(apdiv.lastChild);
        }
        apdiv.appendChild(lorentzian_div);
    }

    get_params();
};


function modelOptions(){
    let fitdiv = document.getElementById("fitdiv")

    fitdiv.innerHTML = `
        <a>
            Peak Width Lower Bound
            <span id="peakWidthLower">2.0</span>
            <input type="range" min="0.0" max="15.0" value="2.0" step="0.1" class="slider"
              id="sliderPeakWidthLower" autocomplete="off"">

            Peak Width Upper Bound
            <span id="peakWidthUpper">12.0</span>
            <input type="range" min="0.0" max="15.0" value="12.0" step="0.1" class="slider"
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


// Fit spectrum
// function onReaderLoad(event){
//     var obj = JSON.parse(event.target.result);

//     // Alloate memory
//     const size = obj.powers.length;
//     const freqs_ptr = get_vec_pointer(size);
//     const freqs_arr = get_array(freqs_ptr, size);

//     const powers_ptr = get_vec_pointer(size);
//     const powers_arr = get_array(powers_ptr, size);

//     // Move js array into shared memory
//     //   so that it's accessible from rust
//     fill_pointer(powers_arr, obj.powers)
//     fill_pointer(freqs_arr, obj.freqs)

//     // Pass to Rust via WASM
//     let powers_fit = fit(freqs_ptr, powers_ptr, size);
// }

// function onChange(event) {
//     var reader = new FileReader();
//     reader.onload = onReaderLoad;
//     reader.readAsText(event.target.files[0]);
// }


//const test = document.getElementById("chooseFile").addEventListener("change", onChange);

// const ap_params_ptr = get_vec_pointer(3);
// const ap_params_arr = get_array(ap_params_ptr, 3);
// fill_pointer(ap_params_arr, [10.0, 2.0, 1.0])

// const pe_params_ptr = get_vec_pointer(3);
// const pe_params_arr = get_array(pe_params_ptr, 3);
// fill_pointer(pe_params_arr, [10.0, 2.0, 0.2])


// let test_sig = simulate(ap_params_ptr, 3, pe_params_ptr, 3, 1.0)