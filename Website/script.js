const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('overlay');
const darkModeToggle = document.getElementById('dark-mode-toggle');
const messageInput = document.getElementById('input-message');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');

// Toggle sidebar
function toggleSidebar() {
    sidebar.classList.toggle('active');
    overlay.classList.toggle('active');
}

menuToggle.addEventListener('click', toggleSidebar);
overlay.addEventListener('click', toggleSidebar);

// Dark mode toggle
darkModeToggle.addEventListener('change', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
});

// Load dark mode setting
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
    darkModeToggle.checked = true;
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (message) {
        addMessage('user', message); // Display user's message
        messageInput.value = ''; // Clear the input field

        try {
            // Update the endpoint to point to your FastAPI server
            const response = await fetch('http://localhost:8000/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: message,
                    max_length: 64,
                    temperature: 0.7
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }

            const data = await response.json();
            // Extract the generated text from the FastAPI response
            const botResponse = data.generated_texts ? data.generated_texts[0] : 'Sorry, I could not generate a response.';
            addMessage('bot', botResponse);
        } catch (error) {
            console.error('Error:', error);
            addMessage('bot', 'Sorry, something went wrong!');
        }
    }
}

// Add this function after the sendMessage function
async function requestJoke() {
    addMessage('user', 'Tell me a joke'); // Display user's request
    
    try {
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: 'tell me a joke',
                max_length: 128,
                temperature: 0.7
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }

        const data = await response.json();
        const botResponse = data.generated_texts ? data.generated_texts[0] : 'Sorry, I could not get a joke right now.';
        addMessage('bot', botResponse);
    } catch (error) {
        console.error('Error:', error);
        addMessage('bot', 'Sorry, something went wrong!');
    }
}

// Add this function after the requestJoke function
async function requestQuote() {
    addMessage('user', 'Give me an inspirational quote'); // Display user's request
    
    try {
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: 'give me a quote',
                max_length: 128,
                temperature: 0.7
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }

        const data = await response.json();
        const botResponse = data.generated_texts ? data.generated_texts[0] : 'Sorry, I could not get a quote right now.';
        addMessage('bot', botResponse);
    } catch (error) {
        console.error('Error:', error);
        addMessage('bot', 'Sorry, something went wrong!');
    }
}

function addMessage(sender, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.textContent = text;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
});

// Initial bot message
addMessage('bot', "Hello! How can I help you today?");


document.getElementById('send-button').addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});


// python -m http.server 3000
// python -m http.server 3000