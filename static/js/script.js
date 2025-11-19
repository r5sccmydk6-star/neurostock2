document.addEventListener("DOMContentLoaded", () => {
    const runBtn = document.getElementById('runBtn'), loader = document.getElementById('loader'), input = document.getElementById('tickerSearch'), drop = document.getElementById('autocomplete-dropdown');
    const currToggle = document.getElementById('currToggle');
    let chart = null, indChart = null, currentData = null, tickers = [], inrRate = 84.0;

    fetch('/api/tickers').then(r=>r.json()).then(d=>{ tickers = d; });

    input.addEventListener('input', (e) => {
        const v = e.target.value.toUpperCase();
        drop.innerHTML = '';
        if(v.length < 1) { drop.style.display = 'none'; return; }
        const m = tickers.filter(t => t.symbol.includes(v) || t.name.toUpperCase().includes(v));
        if(m.length > 0) {
            drop.style.display = 'block';
            m.slice(0,5).forEach(t => {
                const d = document.createElement('div');
                d.className = 'autocomplete-item';
                d.innerHTML = `<span class="item-sym">${t.symbol}</span> <span class="item-name">${t.name}</span>`;
                d.onclick = () => { input.value = t.symbol; drop.style.display = 'none'; };
                drop.appendChild(d);
            });
        } else drop.style.display = 'none';
    });
    document.addEventListener('click', (e) => { if(!e.target.closest('.search-wrapper')) drop.style.display = 'none'; });

    if(runBtn) runBtn.addEventListener('click', async () => {
        loader.classList.add('active');
        try {
            const res = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ticker:input.value||'AAPL', seq_len:document.getElementById('seqLen').value, horizon:document.getElementById('horizon').value}) });
            const data = await res.json();
            if(data.status === 'success') { currentData = data; inrRate = data.inr_rate; updateUI(data); document.getElementById('exportBtn').style.display='inline-flex'; }
            else alert(data.message);
        } catch(e){console.error(e);} finally { loader.classList.remove('active'); }
    });

    if(currToggle) currToggle.addEventListener('change', () => { if(currentData) updateUI(currentData); });

    function updateUI(d) {
        const isINR = currToggle.checked, rate = isINR ? inrRate : 1, sym = isINR ? '₹' : '$';
        document.getElementById('currentPrice').innerText = `${sym}${(d.current_price * rate).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('predictedPrice').innerText = `${sym}${(d.predicted_price * rate).toLocaleString(undefined, {minimumFractionDigits: 2})}`;

        const roi = ((d.predicted_price - d.current_price)/d.current_price)*100;
        const roiEl = document.getElementById('roiValue');
        roiEl.innerText = (roi >= 0 ? '+' : '') + roi.toFixed(2) + '%';
        roiEl.style.color = roi >= 0 ? '#5CFF8D' : '#FF6B6B';

        const sigEl = document.getElementById('neuroSignal');
        sigEl.innerText = d.neuro_score.signal;
        document.getElementById('neuroScoreVal').innerText = `Confidence: ${d.neuro_score.score}%`;

        if(d.neuro_score.color === 'success') sigEl.className = 'kpi-value sig-buy';
        else if(d.neuro_score.color === 'danger') sigEl.className = 'kpi-value sig-sell';
        else sigEl.className = 'kpi-value sig-wait';

        const gap = d.forecast.upper[d.forecast.upper.length-1] - d.forecast.lower[d.forecast.lower.length-1];
        document.getElementById('uncertaintyGap').innerText = '±' + ((gap/d.predicted_price)*100).toFixed(1) + '%';

        if(d.fundamentals) {
            document.getElementById('fundCap').innerText = fmt(d.fundamentals.marketCap);
            document.getElementById('fundPE').innerText = d.fundamentals.peRatio !== 'N/A' ? parseFloat(d.fundamentals.peRatio).toFixed(2) : '-';
            document.getElementById('fundSector').innerText = d.fundamentals.sector;
            document.getElementById('fundHigh').innerText = '$'+d.fundamentals.high52;
        }
        toggleInd('price');
    }

    window.toggleInd = (type) => {
        if(type === 'price') { document.getElementById('priceChartContainer').style.display = 'block'; document.getElementById('indChartContainer').style.display = 'none'; renderPriceChart(); }
        else { document.getElementById('priceChartContainer').style.display = 'none'; document.getElementById('indChartContainer').style.display = 'block'; renderIndChart(type); }
    };

    function renderPriceChart() {
        const ctx = document.getElementById('mainChart').getContext('2d');
        if(chart) chart.destroy();
        const dates = currentData.history.dates.concat(currentData.forecast.dates);
        const lastHist = currentData.history.prices[currentData.history.prices.length-1];
        const forecastData = [lastHist, ...currentData.forecast.mean];

        // Validation Line logic (Actual vs Predicted overlap)
        const valData = currentData.history.validation.concat(new Array(currentData.forecast.dates.length).fill(null));

        const padHist = new Array(currentData.forecast.dates.length).fill(null);
        const padFore = new Array(currentData.history.dates.length-1).fill(null);
        const padUpper = new Array(currentData.history.dates.length-1).fill(null);

        let gradHist = ctx.createLinearGradient(0, 0, 0, 400); gradHist.addColorStop(0, 'rgba(77, 163, 255, 0.3)'); gradHist.addColorStop(1, 'rgba(77, 163, 255, 0)');
        let gradFore = ctx.createLinearGradient(0, 0, 0, 400); gradFore.addColorStop(0, 'rgba(92, 255, 141, 0.3)'); gradFore.addColorStop(1, 'rgba(92, 255, 141, 0)');

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    { label:'Actual', data:currentData.history.prices.concat(padHist), borderColor:'#4DA3FF', borderWidth:3, pointRadius:0, tension:0.3, backgroundColor: gradHist, fill: true },
                    { label:'AI Fit', data:valData, borderColor:'#FBA24F', borderWidth:2, pointRadius:0, borderDash:[4,4], tension:0.3, fill:false }, // ORANGE DASHED
                    { label:'Forecast', data:padFore.concat(forecastData), borderColor:'#5CFF8D', borderWidth:3, borderDash:[5,5], pointRadius:0, tension:0.3, backgroundColor: gradFore, fill: true },
                    { label:'Upper', data:padUpper.concat([lastHist, ...currentData.forecast.upper]), borderColor:'transparent', backgroundColor:'rgba(92, 255, 141, 0.1)', fill:'+1', pointRadius:0 },
                    { label:'Lower', data:padUpper.concat([lastHist, ...currentData.forecast.lower]), borderColor:'transparent', fill:false, pointRadius:0 }
                ]
            },
            options: {
                responsive:true, maintainAspectRatio:false, interaction:{intersect:false, mode:'index'},
                plugins:{legend:{labels:{color:'#aaa'}}, tooltip:{backgroundColor:'rgba(30,30,35,0.9)', titleColor:'#fff', bodyColor:'#ddd', borderColor:'rgba(255,255,255,0.1)', borderWidth:1}},
                scales:{x:{display:false}, y:{grid:{color:'rgba(255,255,255,0.05)'}, ticks:{color:'#888'}}}
            }
        });
    }

    function renderIndChart(type) {
        const ctx = document.getElementById('indicatorChart').getContext('2d');
        if(indChart) indChart.destroy();
        let datasets = [];
        if(type === 'rsi') { datasets = [{ label:'RSI', data:currentData.history.rsi, borderColor:'#FBA24F', borderWidth:2, pointRadius:0 }]; }
        else { datasets = [{ label:'MACD', data:currentData.history.macd, borderColor:'#4DA3FF', borderWidth:2, pointRadius:0 }]; }
        indChart = new Chart(ctx, { type: 'line', data: { labels: currentData.history.dates, datasets: datasets }, options: { responsive:true, maintainAspectRatio:false, scales:{x:{display:false}, y:{grid:{color:'rgba(255,255,255,0.05)'}, ticks:{color:'#888'}}} } });
    }

    function fmt(n) { if(!n) return '-'; if(n>=1e12) return (n/1e12).toFixed(2)+'T'; if(n>=1e9) return (n/1e9).toFixed(2)+'B'; return n; }

    if(exportBtn) { exportBtn.addEventListener('click', () => { if (!currentData) return; let csv = "Date,Forecast_Mean\n"; currentData.forecast.dates.forEach((d, i) => csv += `${d},${currentData.forecast.mean[i]}\n`); const blob = new Blob([csv], { type: 'text/csv' }); const url = window.URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `${input.value}_forecast.csv`; document.body.appendChild(a); a.click(); document.body.removeChild(a); }); }
});