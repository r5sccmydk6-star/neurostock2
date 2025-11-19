document.addEventListener("DOMContentLoaded", () => {
    async function load() {
        const res = await fetch('/api/settings/info'); const data = await res.json();
        document.getElementById('dbSize').innerText = data.db_size_mb + ' MB';
        document.getElementById('modelCount').innerText = data.model_count;
        document.getElementById('tickerCount').innerText = data.ticker_count;
    }
    document.getElementById('clearModelsBtn').addEventListener('click', async()=>{ if(confirm('Reset models?')) { await fetch('/api/settings/clear_models', {method:'POST'}); load(); alert('Done'); } });
    document.getElementById('clearDataBtn').addEventListener('click', async()=>{ if(confirm('Purge data?')) { await fetch('/api/settings/clear_data', {method:'POST'}); load(); alert('Done'); } });
    load();
});