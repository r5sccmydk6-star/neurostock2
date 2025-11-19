document.addEventListener("DOMContentLoaded", () => {
    const tableBody = document.getElementById('portfolioTable');
    const addBtn = document.getElementById('addAssetBtn');
    let allocChart = null;
    let inrRate = 84.0;

    async function loadPortfolio() {
        try {
            const res = await fetch('/api/portfolio');
            const data = await res.json();
            if (data.status === 'success') {
                inrRate = data.rate || 84.0;
                renderTable(data.holdings);
                updateSummary(data.summary, data.holdings.length);
                renderChart(data.allocation);
            }
        } catch (e) { console.error(e); }
    }

    function renderTable(holdings) {
        tableBody.innerHTML = '';
        if (!holdings.length) {
            tableBody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:#9ca3af;">Portfolio Empty</td></tr>';
            return;
        }

        holdings.forEach(i => {
            const row = document.createElement('tr');
            const colorClass = i.gain_loss >= 0 ? 'var(--success)' : 'var(--danger)';
            const sign = i.gain_loss >= 0 ? '+' : '';

            row.innerHTML = `
                <td style="display:flex; align-items:center; gap:10px;">
                    <div style="width:28px; height:28px; background:var(--glass-highlight); border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:0.7rem; color:white;">${i.ticker[0]}</div>
                    <span style="font-weight:bold; color:white;">${i.ticker}</span>
                </td>
                <td>${i.shares}</td>
                <td style="color:#9ca3af">$${i.avg_price.toFixed(2)}</td>
                <td style="font-weight:600; color:white;">$${i.current_price.toFixed(2)}</td>
                <td style="font-weight:bold; color:white;">$${i.total_value.toLocaleString(undefined, {minimumFractionDigits: 0})}</td>
                <td style="color:${colorClass}; font-weight:600;">${sign}$${Math.abs(i.gain_loss).toLocaleString(undefined, {minimumFractionDigits: 0})}</td>
                <td style="text-align:right;"><button onclick="del(${i.id})" class="btn-delete"><i class="fa-solid fa-trash"></i></button></td>
            `;
            tableBody.appendChild(row);
        });
    }

    function updateSummary(s, count) {
        const usdVal = s.total_value;
        const inrVal = usdVal * inrRate;
        document.getElementById('totalValue').innerHTML = `$${usdVal.toLocaleString(undefined, {minimumFractionDigits: 2})} <span style="font-size:0.9rem; color:#888;">₹${inrVal.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>`;
        const g = document.getElementById('totalGain');
        const sign = s.total_gain >= 0 ? '+' : '';
        g.innerText = `${sign}$${Math.abs(s.total_gain).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        g.style.color = s.total_gain >= 0 ? 'var(--success)' : 'var(--danger)';
        document.getElementById('assetCount').innerText = count;
    }

    function renderChart(data) {
        const ctx = document.getElementById('allocationChart').getContext('2d');
        if (allocChart) allocChart.destroy();
        if (!data.data.length) return;
        allocChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{ data: data.data, backgroundColor: ['#4DA3FF', '#5CFF8D', '#FBA24F', '#FF6B6B', '#8b5cf6'], borderWidth: 0, hoverOffset: 4 }]
            },
            options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'right', labels: { color: '#9ca3af', font: {size: 11}, boxWidth: 10, usePointStyle: true } } } }
        });
    }

    addBtn.addEventListener('click', async () => {
        const t = document.getElementById('pTicker').value.trim(), s = parseFloat(document.getElementById('pShares').value), p = parseFloat(document.getElementById('pPrice').value);
        if(!t || isNaN(s) || isNaN(p)) { alert("Please check fields."); return; }
        await fetch('/api/portfolio/add', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ ticker: t.toUpperCase(), shares: s, price: p }) });
        loadPortfolio();
        document.getElementById('pTicker').value=''; document.getElementById('pShares').value=''; document.getElementById('pPrice').value='';
    });

    window.del = async (id) => {
        if(confirm("Delete Asset?")) { await fetch('/api/portfolio/delete', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({id})}); loadPortfolio(); }
    };
    loadPortfolio();
});