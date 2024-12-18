// Function to fetch recognized names and update the webpage
function updateVerifiedNames() {
  // Fetch recognized names from the backend
  fetch('/get_recognized_names')
      .then(response => response.json())
      .then(data => {
          // Get the verified names section element
          const verifiedNamesElement = document.getElementById('verified-names');

          // Split and limit to at most 3 names
          const names = data.names.split(', ').slice(0, 3); // Adjust splitting if needed

          // Update the content of the verified names section
          if (names.length > 0 && names[0] !== "") {
              verifiedNamesElement.textContent = names.join(', '); // Display names as comma-separated string
          } else {
              verifiedNamesElement.textContent = "No one detected"; // Fallback if no names
          }
      })
      .catch(error => {
          console.error('Error fetching recognized names:', error);

          // Update verified names section with an error message
          const verifiedNamesElement = document.getElementById('verified-names');
          verifiedNamesElement.textContent = "Error fetching recognized names";
      });
}

// Call the function at regular intervals (e.g., every 1 second)
setInterval(updateVerifiedNames, 1000);

// Function to fetch and update class info
async function updateClassInfo() {
    try {
      const response = await fetch('/get-class-info');
      const data = await response.json();
  
      // Update the Class Info section dynamically
      document.getElementById('class-info').innerHTML = `
        <strong>Start Time:</strong> ${data.start_time} <br>
        <strong>End Time:</strong> ${data.end_time} <br>
        <strong>Course:</strong> ${data.course_name} <br>
        <strong>Instructor:</strong> ${data.instructor}
      `;
    } catch (error) {
      console.error('Error fetching class info:', error);
      document.getElementById('class-info').innerText = 'Error loading class information.';
    }
  }
  
  // Initial call to update class info
  updateClassInfo();
  
  // Update class info every 60 minutes
  setInterval(updateClassInfo, 10 * 60 * 1000); // 1 hour
  
