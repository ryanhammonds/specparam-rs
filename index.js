import { get_array, get_vec_pointer, fit } from "specparam-rs";

function fill_pointer(arr, values) {
    for (let i = 0; i < values.length; i++){
        arr[i] = values[i];
    }
}

// Fit spectrum
function onReaderLoad(event){
    var obj = JSON.parse(event.target.result);

    // Alloate memory
    const size = obj.powers.length;
    const freqs_ptr = get_vec_pointer(size);
    const freqs_arr = get_array(freqs_ptr, size);

    const powers_ptr = get_vec_pointer(size);
    const powers_arr = get_array(powers_ptr, size);

    // Move js array into shared memory
    //   so that it's accessible from rust
    fill_pointer(powers_arr, obj.powers)
    fill_pointer(freqs_arr, obj.freqs)

    // Pass to Rust via WASM
    let powers_fit = fit(freqs_ptr, powers_ptr, size);

    // Plot
    var graphDiv = document.getElementById('myDiv')
    var trace1 = {
        x: obj.freqs,
        y: obj.powers,
        mode: 'lines',
        name: 'original'
    };
    var trace2 = {
        x: obj.freqs,
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
            autorange: true
        },
        yaxis: {
            type: 'log',
            autorange: true
        }
    };
    var data = [trace1, trace2];
    Plotly.newPlot(graphDiv, data, layout);
}

function onChange(event) {

    var reader = new FileReader();
    reader.onload = onReaderLoad;
    reader.readAsText(event.target.files[0]);


    //var jsonObj = JSON.parse(event.target.result);
    //console.log(jsonObj);


}



const test = document.getElementById("chooseFile").addEventListener("change", onChange);
