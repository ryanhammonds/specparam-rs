"use strict";(self.webpackChunkcreate_wasm_app=self.webpackChunkcreate_wasm_app||[]).push([[10],{420:(e,n,t)=>{t.a(e,(async(e,i)=>{try{t.d(n,{hM:()=>u.hM,m7:()=>u.m7,wg:()=>u.wg});var r=t(316),u=t(980),l=e([r]);r=(l.then?(await l)():l)[0],(0,u.oT)(r),i()}catch(e){i(e)}}))},980:(e,n,t)=>{let i;function r(e){i=e}t.d(n,{$3:()=>A,Bj:()=>F,CN:()=>te,E$:()=>D,F:()=>j,Fr:()=>K,H6:()=>ee,Nl:()=>Q,Od:()=>S,Or:()=>ue,PY:()=>ie,Qz:()=>Y,SU:()=>R,Sc:()=>L,TB:()=>ne,TE:()=>U,To:()=>N,Vb:()=>H,Wc:()=>C,Wl:()=>k,XP:()=>Z,Z$:()=>M,Zf:()=>X,c7:()=>V,cU:()=>B,eY:()=>P,ey:()=>z,fY:()=>re,fr:()=>J,gj:()=>$,h4:()=>T,hM:()=>w,m7:()=>v,m_:()=>q,o$:()=>W,oH:()=>le,oT:()=>r,pT:()=>E,rU:()=>G,ug:()=>O,wg:()=>x}),e=t.hmd(e);const u=new Array(128).fill(void 0);u.push(void 0,null,!0,!1);let l=u.length;function o(e){l===u.length&&u.push(u.length+1);const n=l;return l=u[n],u[n]=e,n}function a(e){return u[e]}function s(e){const n=a(e);return function(e){e<132||(u[e]=l,l=e)}(e),n}let c=new("undefined"==typeof TextDecoder?(0,e.require)("util").TextDecoder:TextDecoder)("utf-8",{ignoreBOM:!0,fatal:!0});c.decode();let d=null;function p(){return null!==d&&0!==d.byteLength||(d=new Uint8Array(i.memory.buffer)),d}function f(e,n){return e>>>=0,c.decode(p().subarray(e,e+n))}function h(e){const n=typeof e;if("number"==n||"boolean"==n||null==e)return`${e}`;if("string"==n)return`"${e}"`;if("symbol"==n){const n=e.description;return null==n?"Symbol":`Symbol(${n})`}if("function"==n){const n=e.name;return"string"==typeof n&&n.length>0?`Function(${n})`:"Function"}if(Array.isArray(e)){const n=e.length;let t="[";n>0&&(t+=h(e[0]));for(let i=1;i<n;i++)t+=", "+h(e[i]);return t+="]",t}const t=/\[object ([^\]]+)\]/.exec(toString.call(e));let i;if(!(t.length>1))return toString.call(e);if(i=t[1],"Object"==i)try{return"Object("+JSON.stringify(e)+")"}catch(e){return"Object"}return e instanceof Error?`${e.name}: ${e.message}\n${e.stack}`:i}let m=0,_=new("undefined"==typeof TextEncoder?(0,e.require)("util").TextEncoder:TextEncoder)("utf-8");const g="function"==typeof _.encodeInto?function(e,n){return _.encodeInto(e,n)}:function(e,n){const t=_.encode(e);return n.set(t),{read:e.length,written:t.length}};let y=null;function b(){return null!==y&&0!==y.byteLength||(y=new Int32Array(i.memory.buffer)),y}function v(e,n,t,r,u,l,o,a,c,d,p,f){return s(i.simulate(e,n,t,r,u,l,o,a,c,d,p,f))}function w(e){return i.get_vec_pointer(e)}function x(e,n){return s(i.get_array(e,n))}function I(e,n){try{return e.apply(this,n)}catch(e){i.__wbindgen_exn_store(o(e))}}function E(e){return o(e)}function O(e){s(e)}function T(e,n){return o(f(e,n))}function M(e){return a(e).now()}function L(e){return o(a(e).crypto)}function k(e){const n=a(e);return"object"==typeof n&&null!==n}function H(e){return o(a(e).process)}function B(e){return o(a(e).versions)}function F(e){return o(a(e).node)}function P(e){return"string"==typeof a(e)}function $(e){return o(a(e).msCrypto)}function C(){return I((function(){return o(e.require)}),arguments)}function W(e){return"function"==typeof a(e)}function U(){return I((function(e,n){a(e).getRandomValues(a(n))}),arguments)}function j(){return I((function(e,n){a(e).randomFillSync(s(n))}),arguments)}function K(){return o(new Array)}function A(e,n){return o(new Function(f(e,n)))}function N(){return I((function(e,n){return o(Reflect.get(a(e),a(n)))}),arguments)}function S(){return I((function(e,n){return o(a(e).call(a(n)))}),arguments)}function q(e){return o(a(e))}function z(){return I((function(){return o(self.self)}),arguments)}function Y(){return I((function(){return o(window.window)}),arguments)}function D(){return I((function(){return o(globalThis.globalThis)}),arguments)}function V(){return I((function(){return o(t.g.global)}),arguments)}function Z(e){return void 0===a(e)}function R(e,n){return a(e).push(a(n))}function Q(){return I((function(e,n,t){return o(a(e).call(a(n),a(t)))}),arguments)}function X(e){return o(a(e).buffer)}function J(e,n,t){return o(new Uint8Array(a(e),n>>>0,t>>>0))}function G(e){return o(new Uint8Array(a(e)))}function ee(e,n,t){a(e).set(a(n),t>>>0)}function ne(e,n,t){return o(new Float64Array(a(e),n>>>0,t>>>0))}function te(e){return o(new Uint8Array(e>>>0))}function ie(e,n,t){return o(a(e).subarray(n>>>0,t>>>0))}function re(e,n){const t=function(e,n,t){if(void 0===t){const t=_.encode(e),i=n(t.length,1)>>>0;return p().subarray(i,i+t.length).set(t),m=t.length,i}let i=e.length,r=n(i,1)>>>0;const u=p();let l=0;for(;l<i;l++){const n=e.charCodeAt(l);if(n>127)break;u[r+l]=n}if(l!==i){0!==l&&(e=e.slice(l)),r=t(r,i,i=l+3*e.length,1)>>>0;const n=p().subarray(r+l,r+i);l+=g(e,n).written}return m=l,r}(h(a(n)),i.__wbindgen_malloc,i.__wbindgen_realloc),r=m;b()[e/4+1]=r,b()[e/4+0]=t}function ue(e,n){throw new Error(f(e,n))}function le(){return o(i.memory)}},10:(e,n,t)=>{t.a(e,(async(e,i)=>{try{t.r(n);var r=t(420),u=e([r]);function l(e,n){for(let t=0;t<n.length;t++)e[t]=n[t]}function o(){var e=[];"undefined"!=typeof sliderInputExp0?(e.push(parseFloat(sliderInputKnee0.value)),e.push(parseFloat(sliderInputExp0.value)),e.push(parseFloat(Math.pow(10,sliderInputOffset0.value))),e.push(parseFloat(sliderInputKnee1.value)),e.push(parseFloat(sliderInputExp1.value)),e.push(parseFloat(Math.pow(10,sliderInputOffset1.value)))):"undefined"!=typeof sliderInputExp&&("undefined"!=typeof sliderInputKnee?(e.push(parseFloat(sliderInputKnee.value)),e.push(parseFloat(sliderInputExp.value)),e.push(parseFloat(Math.pow(10,sliderInputOffset.value)))):(e.push(parseFloat(sliderInputExp.value)),e.push(parseFloat(sliderInputOffset.value))));var n=[];if("undefined"!=typeof sliderOsc0CF&&noscin.value>0){let e=document.getElementById("pediv").children[2].children[0].children;for(let t=0;t<e.length;t++)"input"==e[t].localName&&n.push(parseFloat(e[t].value))}let t=(0,r.hM)(e.length);l((0,r.wg)(t,e.length),e);let i=(0,r.hM)(n.length);l((0,r.wg)(i,n.length),n);let u=document.getElementById("fresin").value,o=document.getElementById("noisein").value,a=0;a="Linear"==selectModel.value?0:"Lorentzian"==selectModel.value?1:2;let s=(0,r.m7)(t,e.length,i,n.length,o,sliderPeakWidthLower.value,sliderPeakWidthUpper.value,sliderMaxPeaks.value,sliderMinHeight.value,sliderPeakThresh.value,a,u),c=s.length/3,d=s.slice(0,c),p=s.slice(c,2*c),f=s.slice(2*c,3*c);var h=document.getElementById("graph"),m=[{x:d,y:p,mode:"lines",name:"original"},{x:d,y:f,mode:"lines",line:{dash:"dash"},name:"fit"}];Plotly.newPlot(h,m,{xaxis:{type:"log"},yaxis:{type:"log"}})}function a(e){let n=document.createElement("div");const t=document.getElementById("pediv");for(;t.childNodes.length>4;)t.removeChild(t.lastChild);if(e>0){let t="<a>";for(let n=0;n<e;n++)t+=`\n              Oscillation ${n}\n                <br>\n                Center Frequency <span id="osc${n}cf">${10*(n+1)}</span>\n                <input type="range" min="1" max="100.0" value="${10*(n+1)}" step="0.1" class="slider"\n                id="sliderOsc${n}CF" autocomplete="off">\n\n                Height <span id="osc${n}PW">${3/(n+1)}</span>\n                <input type="range" min="0" max="10.0" value="${3/(n+1)}" step="0.1" class="slider"\n                id="sliderOsc${n}PW"  autocomplete="off">\n\n                Bandwith <span id="osc${n}bw">1.0</span>\n                <input type="range" min=".1" max="10.0" value="1.0" step="0.1" class="slider"\n                id="sliderOsc${n}BW"  autocomplete="off">\n            `;t+="</a>",n.innerHTML=t;for(let e=0;e<n.children[0].children.length;e++)"input"==n.children[0].children[e].localName&&(n.children[0].children[e].onmouseup=function(){n.children[0].children[e-1].innerHTML=this.value,o()});document.getElementById("pediv").appendChild(n)}o()}function s(){var e=document.getElementById("selectboxap");if("Linear"==e.value){const e=document.createElement("div");e.innerHTML='\n          <a>\n            Exponent\n            <span id="expOutput">2.0</span>\n            <input type="range" min="0.0" max="4.0" value="2.0" step="0.1" class="slider"\n              id="sliderInputExp" autocomplete="off">\n\n            Offset\n            <span id="offOutput">0.0</span>\n            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"\n              id="sliderInputOffset" autocomplete="off">\n          </a>\n        ',e.children[0].children.sliderInputExp.onmouseup=function(){e.children[0].children.expOutput.innerHTML=this.value,o()},e.children[0].children.sliderInputOffset.onmouseup=function(){e.children[0].children.offOutput.innerHTML=this.value,o()};const n=document.getElementById("apdiv");for(;n.childNodes.length>2;)n.removeChild(n.lastChild);n.appendChild(e)}else if("Lorentzian"==e.value){const e=document.createElement("div");e.innerHTML='\n        <a>\n            Knee Frequency\n            <span id="kneeOutput">10.0</span>\n            <input type="range" min="1.0" max="100.0" value="10.0" step="0.1" class="slider"\n              id="sliderInputKnee" autocomplete="off"">\n\n            Exponent\n            <span id="expOutput">2.0</span>\n            <input type="range" min="0.0" max="4.0" value="2.0" step="0.1" class="slider"\n              id="sliderInputExp" autocomplete="off"">\n\n            Offset\n            <span id="offOutput">0.0</span>\n            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"\n              id="sliderInputOffset" autocomplete="off"">\n        </a>\n        ',e.children[0].children.sliderInputKnee.onmouseup=function(){e.children[0].children.kneeOutput.innerHTML=this.value,o()},e.children[0].children.sliderInputExp.onmouseup=function(){e.children[0].children.expOutput.innerHTML=this.value,o()},e.children[0].children.sliderInputOffset.onmouseup=function(){e.children[0].children.offOutput.innerHTML=this.value,o()};const n=document.getElementById("apdiv");for(;n.childNodes.length>2;)n.removeChild(n.lastChild);n.appendChild(e)}else{const e=document.createElement("div");e.innerHTML='\n        <a>\n            Knee Frequency 0\n            <span id="kneeOutput0">10.0</span>\n            <input type="range" min="1.0" max="100.0" value="10.0" step="0.1" class="slider"\n              id="sliderInputKnee0" autocomplete="off"">\n\n            Exponent 0\n            <span id="expOutput0">3.0</span>\n            <input type="range" min="0.0" max="4.0" value="3.0" step="0.1" class="slider"\n              id="sliderInputExp0" autocomplete="off"">\n\n            Offset 0\n            <span id="offOutput0">0.0</span>\n            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"\n              id="sliderInputOffset0" autocomplete="off"">\n\n            Knee Frequency 1\n            <span id="kneeOutput1">50.0</span>\n            <input type="range" min="1.0" max="100.0" value="50.0" step="0.1" class="slider"\n            id="sliderInputKnee1" autocomplete="off"">\n\n            Exponent 1\n            <span id="expOutput1">0.5</span>\n            <input type="range" min="0.0" max="4.0" value="0.5" step="0.1" class="slider"\n            id="sliderInputExp1" autocomplete="off"">\n\n            Offset 1\n            <span id="offOutput1">0.0</span>\n            <input type="range" min="-2.0" max="2.0" value="0.0" step="0.1" class="slider"\n            id="sliderInputOffset1" autocomplete="off"">\n        </a>\n        ',e.children[0].children.sliderInputKnee0.onmouseup=function(){e.children[0].children.kneeOutput0.innerHTML=this.value,o()},e.children[0].children.sliderInputExp0.onmouseup=function(){e.children[0].children.expOutput0.innerHTML=this.value,o()},e.children[0].children.sliderInputOffset0.onmouseup=function(){e.children[0].children.offOutput0.innerHTML=this.value,o()},e.children[0].children.sliderInputKnee1.onmouseup=function(){e.children[0].children.kneeOutput1.innerHTML=this.value,o()},e.children[0].children.sliderInputExp1.onmouseup=function(){e.children[0].children.expOutput1.innerHTML=this.value,o()},e.children[0].children.sliderInputOffset1.onmouseup=function(){e.children[0].children.offOutput1.innerHTML=this.value,o()};const n=document.getElementById("apdiv");for(;n.childNodes.length>2;)n.removeChild(n.lastChild);n.appendChild(e)}o()}function c(){let e=document.getElementById("fitdiv");e.innerHTML='\n        <a>\n            Peak Width Lower Bound\n            <span id="peakWidthLower">0.0</span>\n            <input type="range" min="0.0" max="15.0" value="0.0" step="0.1" class="slider"\n              id="sliderPeakWidthLower" autocomplete="off"">\n\n            Peak Width Upper Bound\n            <span id="peakWidthUpper">12.0</span>\n            <input type="range" min="0.0" max="20.0" value="12.0" step="0.1" class="slider"\n              id="sliderPeakWidthUpper" autocomplete="off"">\n\n            Max Peaks\n            <span id="maxPeaks">0.0</span>\n            <input type="range" min="0.0" max="10.0" value="0.0" step="1.0" class="slider"\n              id="sliderMaxPeaks" autocomplete="off"">\n\n            Min Peak Height\n            <span id="minHeight">0.0</span>\n            <input type="range" min="0.0" max="10.0" value="0.0" step="0.1" class="slider"\n              id="sliderMinHeight" autocomplete="off"">\n\n            Peak Threshold\n            <span id="peakThresh">2.0</span>\n            <input type="range" min="0.0" max="6.0" value="2.0" step="0.1" class="slider"\n              id="sliderPeakThresh" autocomplete="off"">\n\n            Aperiodic Mode\n            <select id="selectModel" autocomplete="off">\n              <option value="Linear" selected="selected">Linear</option>\n              <option value="Lorentzian">Lorentzian</option>\n              <option value="Double Lorentzian">Double Lorentzian</option>\n        </a>\n        ',e.children[0].children.sliderPeakWidthLower.onmouseup=function(){e.children[0].children.peakWidthLower.innerHTML=this.value,o()},e.children[0].children.sliderPeakWidthUpper.onmouseup=function(){e.children[0].children.peakWidthUpper.innerHTML=this.value,o()},e.children[0].children.sliderMaxPeaks.onmouseup=function(){e.children[0].children.maxPeaks.innerHTML=this.value,o()},e.children[0].children.sliderMinHeight.onmouseup=function(){e.children[0].children.minHeight.innerHTML=this.value,o()},e.children[0].children.sliderPeakThresh.onmouseup=function(){e.children[0].children.peakThresh.innerHTML=this.value,o()},document.getElementById("selectModel").addEventListener("change",o)}r=(u.then?(await u)():u)[0],c(),document.getElementById("selectboxap").addEventListener("change",s),document.getElementById("noscin").onmouseup=function(){document.getElementById("nosc").innerHTML=this.value,a(this.value)},document.getElementById("fresin").onmouseup=function(){document.getElementById("fres").innerHTML=this.value,o()},document.getElementById("noisein").onmouseup=function(){document.getElementById("noise").innerHTML=this.value,o()},i()}catch(d){i(d)}}))},316:(e,n,t)=>{var i=t(980);e.exports=t.v(n,e.id,"41fa454dec3767005e88",{"./specparam_bg.js":{__wbindgen_number_new:i.pT,__wbindgen_object_drop_ref:i.ug,__wbindgen_string_new:i.h4,__wbg_now_0cfdc90c97d0c24b:i.Z$,__wbg_crypto_c48a774b022d20ac:i.Sc,__wbindgen_is_object:i.Wl,__wbg_process_298734cf255a885d:i.Vb,__wbg_versions_e2e78e134e3e5d01:i.cU,__wbg_node_1cd7a5d853dbea79:i.Bj,__wbindgen_is_string:i.eY,__wbg_msCrypto_bcb970640f50a1e8:i.gj,__wbg_require_8f08ceecec0f4fee:i.Wc,__wbindgen_is_function:i.o$,__wbg_getRandomValues_37fa2ca9e4e07fab:i.TE,__wbg_randomFillSync_dc1e9a60c158336d:i.F,__wbg_new_898a68150f225f2e:i.Fr,__wbg_newnoargs_581967eacc0e2604:i.$3,__wbg_get_97b561fb56f034b5:i.To,__wbg_call_cb65541d95d71282:i.Od,__wbindgen_object_clone_ref:i.m_,__wbg_self_1ff1d729e9aae938:i.ey,__wbg_window_5f4faef6c12b79ec:i.Qz,__wbg_globalThis_1d39714405582d3c:i.E$,__wbg_global_651f05c6a0944d1c:i.c7,__wbindgen_is_undefined:i.XP,__wbg_push_ca1c26067ef907ac:i.SU,__wbg_call_01734de55d61e11d:i.Nl,__wbg_buffer_085ec1f694018c4f:i.Zf,__wbg_newwithbyteoffsetandlength_6da8e527659b86aa:i.fr,__wbg_new_8125e318e6245eed:i.rU,__wbg_set_5cf90238115182c3:i.H6,__wbg_newwithbyteoffsetandlength_b8047c68e84e60be:i.TB,__wbg_newwithlength_e5d69174d6984cd7:i.CN,__wbg_subarray_13db269f57aa838d:i.PY,__wbindgen_debug_string:i.fY,__wbindgen_throw:i.Or,__wbindgen_memory:i.oH}})}}]);