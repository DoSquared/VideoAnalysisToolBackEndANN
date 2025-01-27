/**
 * Web Vitals Performance Measurement Module
 * This module provides functions to measure various web performance metrics,
 * which are essential for understanding and improving user experience. It uses
 * the PerformanceObserver API to observe performance entry types and captures
 * key metrics such as Cumulative Layout Shift (CLS), First Contentful Paint (FCP),
 * First Input Delay (FID), Largest Contentful Paint (LCP), and Time to First Byte (TTFB).
 *
 * Functions:
 *
 * 1. getCLS (Cumulative Layout Shift)
 *    - Measures the sum of all unexpected layout shifts that occur during the
 *      lifespan of the page. Layout shifts are sudden changes in a web page's layout.
 *
 * 2. getFCP (First Contentful Paint)
 *    - Measures the time from when the page starts loading to when any part of the
 *      page's content is rendered on the screen. It captures the time it takes for
 *      the first piece of DOM content to be painted.
 *
 * 3. getFID (First Input Delay)
 *    - Measures the time from when a user first interacts with the site (e.g., clicks
 *      a link, taps a button) to the time when the browser is actually able to begin
 *      processing event handlers in response to that interaction.
 *
 * 4. getLCP (Largest Contentful Paint)
 *    - Measures the time from when the page starts loading to when the largest
 *      text block or image is rendered on the screen. It captures the loading
 *      performance of the main content.
 *
 * 5. getTTFB (Time to First Byte)
 *    - Measures the time it takes for the browser to receive the first byte of
 *      response from the server after a navigation request. It indicates the
 *      responsiveness of the web server.
 *
 * Usage:
 * These functions can be imported and used in a web application to monitor and
 * report performance metrics. They are particularly useful for optimizing the
 * performance of a website and improving the overall user experience.
 *
 * Example:
 * import { getCLS, getFCP, getFID, getLCP, getTTFB } from 'web-vitals';
 *
 * getCLS(console.log);
 * getFCP(console.log);
 * getFID(console.log);
 * getLCP(console.log);
 * getTTFB(console.log);
 */

"use strict";(self.webpackChunkfrontend=self.webpackChunkfrontend||[]).push([[787],{787:function(e,n,t){t.r(n),t.d(n,{getCLS:function(){return y},getFCP:function(){return g},getFID:function(){return C},getLCP:function(){return P},getTTFB:function(){return D}});var i,r,a,o,u=function(e,n){return{name:e,value:void 0===n?-1:n,delta:0,entries:[],id:"v2-".concat(Date.now(),"-").concat(Math.floor(8999999999999*Math.random())+1e12)}},c=function(e,n){try{if(PerformanceObserver.supportedEntryTypes.includes(e)){if("first-input"===e&&!("PerformanceEventTiming"in self))return;var t=new PerformanceObserver((function(e){return e.getEntries().map(n)}));return t.observe({type:e,buffered:!0}),t}}catch(e){}},f=function(e,n){var t=function t(i){"pagehide"!==i.type&&"hidden"!==document.visibilityState||(e(i),n&&(removeEventListener("visibilitychange",t,!0),removeEventListener("pagehide",t,!0)))};addEventListener("visibilitychange",t,!0),addEventListener("pagehide",t,!0)},s=function(e){addEventListener("pageshow",(function(n){n.persisted&&e(n)}),!0)},m=function(e,n,t){var i;return function(r){n.value>=0&&(r||t)&&(n.delta=n.value-(i||0),(n.delta||void 0===i)&&(i=n.value,e(n)))}},v=-1,d=function(){return"hidden"===document.visibilityState?0:1/0},p=function(){f((function(e){var n=e.timeStamp;v=n}),!0)},l=function(){return v<0&&(v=d(),p(),s((function(){setTimeout((function(){v=d(),p()}),0)}))),{get firstHiddenTime(){return v}}},g=function(e,n){var t,i=l(),r=u("FCP"),a=function(e){"first-contentful-paint"===e.name&&(f&&f.disconnect(),e.startTime<i.firstHiddenTime&&(r.value=e.startTime,r.entries.push(e),t(!0)))},o=window.performance&&performance.getEntriesByName&&performance.getEntriesByName("first-contentful-paint")[0],f=o?null:c("paint",a);(o||f)&&(t=m(e,r,n),o&&a(o),s((function(i){r=u("FCP"),t=m(e,r,n),requestAnimationFrame((function(){requestAnimationFrame((function(){r.value=performance.now()-i.timeStamp,t(!0)}))}))})))},h=!1,T=-1,y=function(e,n){h||(g((function(e){T=e.value})),h=!0);var t,i=function(n){T>-1&&e(n)},r=u("CLS",0),a=0,o=[],v=function(e){if(!e.hadRecentInput){var n=o[0],i=o[o.length-1];a&&e.startTime-i.startTime<1e3&&e.startTime-n.startTime<5e3?(a+=e.value,o.push(e)):(a=e.value,o=[e]),a>r.value&&(r.value=a,r.entries=o,t())}},d=c("layout-shift",v);d&&(t=m(i,r,n),f((function(){d.takeRecords().map(v),t(!0)})),s((function(){a=0,T=-1,r=u("CLS",0),t=m(i,r,n)})))},E={passive:!0,capture:!0},w=new Date,L=function(e,n){i||(i=n,r=e,a=new Date,F(removeEventListener),S())},S=function(){if(r>=0&&r<a-w){var e={entryType:"first-input",name:i.type,target:i.target,cancelable:i.cancelable,startTime:i.timeStamp,processingStart:i.timeStamp+r};o.forEach((function(n){n(e)})),o=[]}},b=function(e){if(e.cancelable){var n=(e.timeStamp>1e12?new Date:performance.now())-e.timeStamp;"pointerdown"==e.type?function(e,n){var t=function(){L(e,n),r()},i=function(){r()},r=function(){removeEventListener("pointerup",t,E),removeEventListener("pointercancel",i,E)};addEventListener("pointerup",t,E),addEventListener("pointercancel",i,E)}(n,e):L(n,e)}},F=function(e){["mousedown","keydown","touchstart","pointerdown"].forEach((function(n){return e(n,b,E)}))},C=function(e,n){var t,a=l(),v=u("FID"),d=function(e){e.startTime<a.firstHiddenTime&&(v.value=e.processingStart-e.startTime,v.entries.push(e),t(!0))},p=c("first-input",d);t=m(e,v,n),p&&f((function(){p.takeRecords().map(d),p.disconnect()}),!0),p&&s((function(){var a;v=u("FID"),t=m(e,v,n),o=[],r=-1,i=null,F(addEventListener),a=d,o.push(a),S()}))},k={},P=function(e,n){var t,i=l(),r=u("LCP"),a=function(e){var n=e.startTime;n<i.firstHiddenTime&&(r.value=n,r.entries.push(e),t())},o=c("largest-contentful-paint",a);if(o){t=m(e,r,n);var v=function(){k[r.id]||(o.takeRecords().map(a),o.disconnect(),k[r.id]=!0,t(!0))};["keydown","click"].forEach((function(e){addEventListener(e,v,{once:!0,capture:!0})})),f(v,!0),s((function(i){r=u("LCP"),t=m(e,r,n),requestAnimationFrame((function(){requestAnimationFrame((function(){r.value=performance.now()-i.timeStamp,k[r.id]=!0,t(!0)}))}))}))}},D=function(e){var n,t=u("TTFB");n=function(){try{var n=performance.getEntriesByType("navigation")[0]||function(){var e=performance.timing,n={entryType:"navigation",startTime:0};for(var t in e)"navigationStart"!==t&&"toJSON"!==t&&(n[t]=Math.max(e[t]-e.navigationStart,0));return n}();if(t.value=t.delta=n.responseStart,t.value<0||t.value>performance.now())return;t.entries=[n],e(t)}catch(e){}},"complete"===document.readyState?setTimeout(n,0):addEventListener("load",(function(){return setTimeout(n,0)}))}}}]);
//# sourceMappingURL=787.cda612ba.chunk.js.map
