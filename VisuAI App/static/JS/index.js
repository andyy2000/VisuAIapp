/* 
Author: Andy Yang
Date: 2024-10-11 (updated 2026-10)
Project: VisuAI Front End
Description: Created the Visuals and Details of VisuAI Homepage with Flask, HTML, CSS, and JavaScript
*/

const sections = [
    { id: 'intro', duration: 1250 }, 
    { id: 'getting-started', duration: 1250 }, 
    { id: 'open-app', duration: 1250 },
    { id: 'arrow1', duration: 1250 },
    { id: 'access-camera', duration: 1250 },
    { id: 'arrow2', duration: 1250 },
    { id: 'wake-word', duration: 1250 },
    { id: 'arrow-down1', duration: 1250 },
    { id: 'using-object-detection', duration: 1250 },
    { id: 'capture-image', duration: 1250 },
    { id: 'arrow3', duration: 1250 },
    { id: 'get-object-info', duration: 1250 },
    { id: 'arrow-down2', duration: 1250 },
    { id: 'asking-questions', duration: 1250 },
    { id: 'voice-commands', duration: 1250 },
    { id: 'arrow4', duration: 1250 },
    { id: 'find-objects', duration: 1250 },
    { id: 'arrow-down3', duration: 1250 },
    { id: 'receiving-descriptions', duration: 1250 },
    { id: 'scene-description', duration: 1250 },
    { id: 'arrow5', duration: 1250 },
    { id: 'emergency-situations', duration: 1250 },
    { id: 'arrow-down4', duration: 1250 },
    { id: 'additional-features', duration: 1250 },
    { id: 'voice-feedback', duration: 1250 },
    { id: 'arrow6', duration: 1250 },
    { id: 'hazard-alerts', duration: 1250 },
    { id: 'troubleshooting', duration: 1250 }
];

let currentIndex = 0;

function highlightNext() {
    document.querySelectorAll('.section, .sub-section, .arrow-right, .arrow-down').forEach(el => {
        el.classList.remove('glow-border');
        el.classList.remove('glow-arrow');
    });

    if (currentIndex >= sections.length) {
        currentIndex = 0;
    }

    const { id, duration } = sections[currentIndex];

    const element = document.getElementById(id);
    if (element) {
        if (element.classList.contains('arrow-right') || element.classList.contains('arrow-down')) {
            element.classList.add('glow-arrow');
        } else {
            element.classList.add('glow-border');
        }
    }

    currentIndex++;

    setTimeout(highlightNext, duration);
}

highlightNext();

const video = document.getElementById('background-video');
if (video) {
    video.playbackRate = 0.5;
}