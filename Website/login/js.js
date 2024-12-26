document.addEventListener('DOMContentLoaded', () => {
    const toggleSwitch = document.querySelector('#dark-mode-toggle');
    const loginButton = document.querySelector('#login-button');

    toggleSwitch.addEventListener('change', () => {
        document.body.classList.toggle('dark-mode');
    });

    loginButton.addEventListener('click', (event) => {
        event.preventDefault(); // Prevent the default form submission
        window.location.href = 'index.html'; // Redirect to the chat window
    });
});
