/* 
Author: Andy Yang
Date: 2024-10-11 (updated 2026-10)
Project: VisuAI Front End
Description: Polls /status and shows the latest scene description, nearby objects,
and detection speed on the camera page.
*/

document.addEventListener('DOMContentLoaded', () => {
    const descriptionEl = document.getElementById('description');
    const objectsEl = document.getElementById('objects');
    const statsEl = document.getElementById('stats');
    const errorEl = document.getElementById('error');

    async function refreshStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            if (data.error) {
                errorEl.textContent = data.error;
                errorEl.hidden = false;
            } else {
                errorEl.hidden = true;
            }

            // Only update when the text changes, so screen readers don't repeat it.
            if (descriptionEl.textContent !== data.description) {
                descriptionEl.textContent = data.description;
            }

            objectsEl.replaceChildren(...data.objects.map(text => {
                const li = document.createElement('li');
                li.textContent = text;
                return li;
            }));

            statsEl.textContent = data.fps ? `Detection running at ${data.fps} frames per second` : '';
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    }

    refreshStatus();
    setInterval(refreshStatus, 2000);
});