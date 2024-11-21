// Function to fetch recognized names and update the webpage
function updateRecognizedNames() {
  fetch('/get_recognized_names')
      .then(response => response.json())
      .then(data => {
          // Update the verified names
          const verifiedNamesElement = document.getElementById('verified-names');
          verifiedNamesElement.textContent = data.names; // Update names as a comma-separated string
      })
      .catch(error => console.error('Error fetching recognized names:', error));
}

// Call the function at regular intervals (e.g., every 1 second)
setInterval(updateRecognizedNames, 1000);


async function fetchClassInfo() {
  try {
    const response = await fetch('/get_class_info'); // Flask endpoint for class info
    const data = await response.json();

    // Update the "Class Info" section
    document.getElementById("class-info").innerText =
      `Class: ${data.class}, Instructor: ${data.instructor}, Time: ${data.time_start} - ${data.time_end}`;
  } catch (error) {
    console.error("Error fetching class info:", error);
    document.getElementById("class-info").innerText = "Error fetching class information.";
  }
}

// Poll the server for class info every minute
setInterval(fetchClassInfo, 60000); // 60,000 ms = 1 minute

// Fetch class info immediately on page load
fetchClassInfo();